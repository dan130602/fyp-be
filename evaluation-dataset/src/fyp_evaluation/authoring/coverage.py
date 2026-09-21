"""Narrow K01 cell decoding; no CSV loader or call compiler is implemented."""

from fyp_evaluation.validation.identification import is_invalid_test_identifier


def parse_k01_cell(cell):
    """Decode K01 only: NOT_STATED -> None; preserve valid test text exactly.

    A future compiler must call this only for CLIENT_KYC K01 with field
    client_identification_number. JSON serialization turns None into null.
    No whitespace, case or checksum correction is performed.
    """
    if cell == "NOT_STATED":
        return None
    if not is_invalid_test_identifier(cell):
        raise ValueError("K01 requires NOT_STATED or a checksum-invalid nine-character test identifier")
    return cell
