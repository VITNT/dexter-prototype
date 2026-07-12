# Install the official WLASL300 checkpoint

1. Read the WLASL Computational Use of Data Agreement at
   `third_party/WLASL/start_kit/C-UDA-1.0.pdf`. WLASL states that the data is
   for academic/computational use and not commercial use.
2. Open the official pretrained-weights link:
   <https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view>
3. Download `archived.zip` and extract it outside the repository.
4. Open its `I3D/archived/asl300/` directory. Prefer the checkpoint whose name
   contains `nslt_300_005100` (the official code marks it as the best result).
   The reported-results checkpoint may contain `nslt_300_005500`.
5. Copy the selected `.pt` file here and rename it exactly to:

   `wlasl300.pt`

The final path must be `asl_service/models/wlasl300.pt`. Do not commit this
large checkpoint to Git.

