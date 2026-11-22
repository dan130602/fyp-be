import threading

import pytest

from meralion_worker import storage as st


# ---- local backend -------------------------------------------------------------------------------------------------------------------------------------------

def test_local_write_then_read_and_exists(tmp_path):
    store = st.LocalStorage(tmp_path)
    assert not store.exists("transcripts/a.json")
    store.write_exclusive("transcripts/a.json", "费用".encode("utf-8"))
    assert store.exists("transcripts/a.json") and store.read("transcripts/a.json") == "费用".encode("utf-8")
    assert (tmp_path / "transcripts" / "a.json").is_file()


def test_local_write_is_exclusive_and_never_overwrites(tmp_path):
    store = st.LocalStorage(tmp_path)
    store.write_exclusive("k.json", b"first")
    with pytest.raises(st.AlreadyExists):
        store.write_exclusive("k.json", b"second")
    assert store.read("k.json") == b"first"


def test_local_write_leaves_no_temporary_files(tmp_path):
    store = st.LocalStorage(tmp_path)
    store.write_exclusive("d/k.json", b"x")
    with pytest.raises(st.AlreadyExists):
        store.write_exclusive("d/k.json", b"y")
    assert sorted(p.name for p in (tmp_path / "d").iterdir()) == ["k.json"]


def test_local_concurrent_writers_produce_exactly_one_winner(tmp_path):
    store = st.LocalStorage(tmp_path)
    results = []

    def attempt(n):
        try:
            store.write_exclusive("plan.json", f"writer-{n}".encode())
            results.append("won")
        except st.AlreadyExists:
            results.append("lost")

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("won") == 1 and results.count("lost") == 7
    assert store.read("plan.json").startswith(b"writer-")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plan.json"]


def test_local_read_of_a_missing_key(tmp_path):
    with pytest.raises(st.ObjectNotFound):
        st.LocalStorage(tmp_path).read("nope.json")


@pytest.mark.parametrize("key", ["", "/abs.json", "../escape.json", "a/../../b.json", "a\\b.json", "C:/x.json"])
def test_unsafe_keys_are_refused(tmp_path, key):
    store = st.LocalStorage(tmp_path)
    with pytest.raises(st.StorageError, match="invalid storage key"):
        store.write_exclusive(key, b"x")
    with pytest.raises(st.StorageError, match="invalid storage key"):
        store.exists(key)


def test_local_download_uses_the_input_in_place(tmp_path):
    source = tmp_path / "in.wav"
    source.write_bytes(b"RIFF")
    assert st.LocalStorage(tmp_path / "out").download(str(source), tmp_path / "scratch") == source
    with pytest.raises(st.ObjectNotFound):
        st.LocalStorage(tmp_path).download(str(tmp_path / "missing.wav"), tmp_path)


# ---- S3 backend (fake client; no network, no boto3 import) ------------------------------------------------------------------------------------------------------

class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self, fail_with=None):
        self.objects, self.calls, self.fail_with = {}, [], fail_with

    def _maybe_fail(self):
        if self.fail_with:
            raise self.fail_with

    def head_object(self, Bucket, Key):
        self.calls.append(("head", Bucket, Key))
        self._maybe_fail()
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {}

    def get_object(self, Bucket, Key):
        self.calls.append(("get", Bucket, Key))
        self._maybe_fail()
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("NoSuchKey")
        data = self.objects[(Bucket, Key)]
        return {"Body": type("Body", (), {"read": lambda self_: data})()}

    def put_object(self, Bucket, Key, Body, ContentType, IfNoneMatch):
        self.calls.append(("put", Bucket, Key, ContentType, IfNoneMatch))
        self._maybe_fail()
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise FakeClientError("PreconditionFailed")
        self.objects[(Bucket, Key)] = Body

    def download_file(self, bucket, key, filename):
        self.calls.append(("download", bucket, key, filename))
        self._maybe_fail()
        if (bucket, key) not in self.objects:
            raise FakeClientError("404")
        open(filename, "wb").write(self.objects[(bucket, key)])


def test_s3_upload_uses_the_key_content_type_and_conditional_write():
    client = FakeS3()
    store = st.S3Storage("out-bucket", "state/inv-1", client=client)
    store.write_exclusive("transcripts/job.json", b"{}")
    assert client.calls == [("put", "out-bucket", "state/inv-1/transcripts/job.json", "application/json", "*")]
    assert store.read("transcripts/job.json") == b"{}"


def test_s3_without_prefix_uses_the_bare_key():
    client = FakeS3()
    st.S3Storage("b", client=client).write_exclusive("transcripts/j.json", b"x")
    assert client.calls[0][2] == "transcripts/j.json"


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_s3_exists_treats_only_missing_codes_as_absent(code):
    class Absent(FakeS3):
        def head_object(self, Bucket, Key):
            raise FakeClientError(code)

    assert st.S3Storage("b", client=Absent()).exists("k.json") is False


def test_s3_exists_true_for_a_present_object():
    client = FakeS3()
    client.objects[("b", "k.json")] = b"x"
    assert st.S3Storage("b", client=client).exists("k.json") is True


@pytest.mark.parametrize("code", ["403", "AccessDenied", "SlowDown", "InternalError", "500"])
def test_s3_permission_and_other_errors_are_raised_never_treated_as_absent(code):
    store = st.S3Storage("b", client=FakeS3(fail_with=FakeClientError(code)))
    with pytest.raises(st.StorageError) as info:
        store.exists("k.json")
    assert info.value.code == code and not isinstance(info.value, st.ObjectNotFound)


def test_s3_errors_without_a_code_such_as_network_failures_are_raised():
    store = st.S3Storage("b", client=FakeS3(fail_with=ConnectionError("network down")))
    with pytest.raises(st.StorageError):
        store.exists("k.json")


@pytest.mark.parametrize("code", ["412", "PreconditionFailed"])
def test_s3_conditional_put_failure_means_already_exists(code):
    class Taken(FakeS3):
        def put_object(self, **kwargs):
            raise FakeClientError(code)

    with pytest.raises(st.AlreadyExists):
        st.S3Storage("b", client=Taken()).write_exclusive("k.json", b"x")


def test_s3_second_exclusive_write_does_not_overwrite():
    client = FakeS3()
    store = st.S3Storage("b", client=client)
    store.write_exclusive("k.json", b"first")
    with pytest.raises(st.AlreadyExists):
        store.write_exclusive("k.json", b"second")
    assert store.read("k.json") == b"first"


def test_s3_write_permission_error_is_not_reported_as_exists():
    with pytest.raises(st.StorageError) as info:
        st.S3Storage("b", client=FakeS3(fail_with=FakeClientError("AccessDenied"))).write_exclusive("k.json", b"x")
    assert not isinstance(info.value, st.AlreadyExists) and info.value.code == "AccessDenied"


def test_s3_read_of_a_missing_object_is_object_not_found():
    with pytest.raises(st.ObjectNotFound):
        st.S3Storage("b", client=FakeS3()).read("nope.json")


def test_s3_download_keeps_the_file_name_and_extension(tmp_path):
    client = FakeS3()
    client.objects[("in-bucket", "uploads/u1/abc-call.mp3")] = b"ID3"
    path = st.S3Storage("out", client=client).download("s3://in-bucket/uploads/u1/abc-call.mp3", tmp_path)
    assert path == tmp_path / "abc-call.mp3" and path.read_bytes() == b"ID3" and path.suffix == ".mp3"


def test_s3_download_of_a_missing_object_and_permission_error(tmp_path):
    with pytest.raises(st.ObjectNotFound):
        st.S3Storage("out", client=FakeS3()).download("s3://in-bucket/missing.wav", tmp_path)
    with pytest.raises(st.StorageError) as info:
        st.S3Storage("out", client=FakeS3(fail_with=FakeClientError("403"))).download("s3://in-bucket/x.wav", tmp_path)
    assert not isinstance(info.value, st.ObjectNotFound)


@pytest.mark.parametrize("uri", ["http://x/y", "s3://bucket", "s3:///key", "bucket/key"])
def test_bad_s3_uris(uri):
    with pytest.raises(st.StorageError):
        st.parse_s3_uri(uri)
