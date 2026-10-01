from pathlib import Path
import subprocess
import zipfile


class MinerUEngine:

    def __init__(
        self,
        pdf_path: str,
        output_dir: str
    ):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)

    # ---------------------------------------------------------
    # Validate PDF
    # ---------------------------------------------------------
    def validate_input(self):

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Input file must be a PDF."
            )

    # ---------------------------------------------------------
    # Prepare output directory
    # ---------------------------------------------------------
    def prepare_output(self):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # Run MinerU Parser
    # ---------------------------------------------------------
    def run(self):

        self.validate_input()
        self.prepare_output()

        zip_path = (
            self.output_dir /
            "mineru_result.zip"
        )

        command = [
            "mineru-kit",
            "parse",

            str(self.pdf_path),

            "-o",
            str(zip_path),

            "--pages",
            "all",

            "--format",
            "zip",

            "--tier",
            "basic"
        ]

        print("=" * 80)
        print("MINERU PDF PARSER")
        print("=" * 80)

        print(f"\nInput : {self.pdf_path}")
        print(f"Output: {self.output_dir}")
        print("Tier  : basic")
        print("VLM   : disabled by using Basic tier")

        print("\nParser command:")
        print(" ".join(command))

        print("\nStarting PDF parsing...\n")

        result = subprocess.run(
            command,
            text=True
        )

        # -----------------------------------------------------
        # Check MinerU result
        # -----------------------------------------------------

        if result.returncode != 0:

            raise RuntimeError(
                "MinerU PDF parsing failed."
            )

        print("\n" + "=" * 80)
        print("MINERU PARSING COMPLETED")
        print("=" * 80)

        # -----------------------------------------------------
        # Extract ZIP package
        # -----------------------------------------------------

        extracted_dir = (
            self.output_dir /
            "extracted"
        )

        extracted_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print("\nExtracting MinerU result package...")

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                extracted_dir
            )

        # -----------------------------------------------------
        # Display extracted files
        # -----------------------------------------------------

        print("\nExtraction package ready.")

        print("\nGenerated files:")

        for path in extracted_dir.rglob("*"):

            if path.is_file():

                print(
                    f"  {path.relative_to(extracted_dir)}"
                )

        return extracted_dir