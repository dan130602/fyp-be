"""Storage for inputs, outputs and resume state. Two backends with the same operations:

    exists(key) -> bool        read(key) -> bytes        write_exclusive(key, data) -> None (AlreadyExists if present; never overwrites)        download(uri, dest_dir) -> Path

Exclusive writes make a plan or an output impossible to overwrite or to write divergently from two processes. They are NOT a guard against two processes doing the same compute at once.
"""
import os
import tempfile
from pathlib import Path, PurePosixPath

ABSENT_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
EXISTS_CODES = frozenset({"412", "PreconditionFailed"})


class StorageError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class AlreadyExists(StorageError):
    pass


class ObjectNotFound(StorageError):
    pass


def _check_key(key):
    path = PurePosixPath(key)
    if not key or key.startswith("/") or "\\" in key or ".." in path.parts or ":" in path.parts[0]:
        raise StorageError(f"invalid storage key: {key!r}")
    return str(path)


def _code_of(error):
    """Error code of a botocore ClientError, by duck typing (no botocore import); None for errors that carry no code (network problems and the like)."""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        return None if code is None else str(code)
    return None


class LocalStorage:
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, key):
        return self.root / _check_key(key)

    def exists(self, key):
        return self._path(key).is_file()

    def read(self, key):
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise ObjectNotFound(f"not found: {key}")

    def write_exclusive(self, key, data):
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".tmp-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)                   # atomic and fails if the destination exists
            except FileExistsError:
                raise AlreadyExists(f"already exists: {key}")
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def download(self, uri, dest_dir):
        """A local input is used where it is: no copy. Returns its path."""
        path = Path(uri)
        if not path.is_file():
            raise ObjectNotFound(f"input not found: {uri}")
        return path


def parse_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise StorageError(f"not an s3 uri: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise StorageError(f"s3 uri needs a bucket and a key: {uri!r}")
    return bucket, key


class S3Storage:
    """Keys are relative to `prefix` inside `bucket`. The client is created lazily (boto3 is imported only when no client is injected).

    write_exclusive uses an S3 conditional write (If-None-Match: *); it needs a boto3 that supports it (the image lock pins one)."""

    def __init__(self, bucket, prefix="", *, region=None, client=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def _key(self, key):
        key = _check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def exists(self, key):
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as error:
            code = _code_of(error)
            if code in ABSENT_CODES:
                return False
            raise StorageError(f"head_object failed for {self._key(key)}: {error}", code) from error      # 403, AccessDenied, throttling, network: never "absent"

    def read(self, key):
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except Exception as error:
            code = _code_of(error)
            if code in ABSENT_CODES:
                raise ObjectNotFound(f"not found: {self._key(key)}", code) from error
            raise StorageError(f"get_object failed for {self._key(key)}: {error}", code) from error

    def write_exclusive(self, key, data, content_type="application/json"):
        try:
            self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data, ContentType=content_type, IfNoneMatch="*")
        except Exception as error:
            code = _code_of(error)
            if code in EXISTS_CODES:
                raise AlreadyExists(f"already exists: {self._key(key)}", code) from error
            raise StorageError(f"put_object failed for {self._key(key)}: {error}", code) from error

    def download(self, uri, dest_dir):
        """Download an input object (any bucket) into dest_dir, keeping its file name and extension."""
        bucket, key = parse_s3_uri(uri)
        destination = Path(dest_dir) / PurePosixPath(key).name
        try:
            self.client.download_file(bucket, key, str(destination))
        except Exception as error:
            code = _code_of(error)
            if code in ABSENT_CODES:
                raise ObjectNotFound(f"not found: {uri}", code) from error
            raise StorageError(f"download failed for {uri}: {error}", code) from error
        return destination
