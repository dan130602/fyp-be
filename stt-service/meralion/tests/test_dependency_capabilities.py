"""The S3 conditional write is what makes outputs and resume state impossible to overwrite. The mock-client tests in test_storage.py cannot show that the installed SDK has it, so check the SDK's own service model."""
import re
from pathlib import Path

import botocore.session

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def put_object_members():
    return botocore.session.get_session().get_service_model("s3").operation_model("PutObject").input_shape.members


def test_the_installed_sdk_supports_the_arguments_storage_py_sends():
    members = put_object_members()
    for argument in ("Bucket", "Key", "Body", "ContentType", "IfNoneMatch"):
        assert argument in members, f"the installed botocore PutObject has no {argument}"


def test_the_sdk_calls_storage_py_makes_exist_in_the_service_model():
    model = botocore.session.get_session().get_service_model("s3")
    for operation in ("HeadObject", "GetObject", "PutObject"):
        assert operation in model.operation_names


def version_of(requirements_file, package):
    for line in (PACKAGE_ROOT / requirements_file).read_text(encoding="utf-8").splitlines():
        match = re.match(rf"^{package}(==|>=)([0-9][0-9.]*)", line.strip())
        if match:
            return match.group(1), tuple(int(part) for part in match.group(2).split("."))
    raise AssertionError(f"{package} not found in {requirements_file}")


def test_the_dev_floor_for_boto3_is_not_below_the_version_the_image_lock_pins():
    """The lock pins the version the S3 behaviour is verified with; the development environment must not allow an older SDK than that."""
    operator, floor = version_of("requirements-dev.txt", "boto3")
    _, locked = version_of("requirements-transformers.lock", "boto3")
    assert operator in (">=", "==") and floor >= locked, f"requirements-dev.txt allows boto3 {floor}, below the locked {locked}"
