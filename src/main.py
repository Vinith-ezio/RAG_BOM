from extraction.mineru_engine import MinerUEngine


INPUT_PDF = (
    r"E:\PDF Ingestion\data\input\BHEL Spec..pdf"
)

OUTPUT_DIR = (
    r"E:\PDF Ingestion\data\output\BHEL_basic"
)


def main():

    engine = MinerUEngine(
        pdf_path=INPUT_PDF,
        output_dir=OUTPUT_DIR
    )

    output = engine.run()

    print("\n" + "=" * 80)
    print("PDF PARSING COMPLETE")
    print("=" * 80)

    print(f"\nParsed output:")
    print(output)


if __name__ == "__main__":
    main()