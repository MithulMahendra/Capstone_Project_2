from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HierarchicalChunker
from docling_core.types.doc import PictureItem


def _clean_cell_value(val) -> str:
    if val is None:
        return ""
    val = str(val).strip()
    if val.lower() in {"nan", "none", "null", "", "-"}:
        return ""
    return val


def parse_and_chunk(file_path: str) -> list[dict]:
    # Configure the pipeline for complex PDFs (OCR + Tables + Pictures)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(file_path)
    doc = result.document

    chunker = HierarchicalChunker()
    chunks = list(chunker.chunk(doc))

    raw_parsed = []
    last_known_headings = []

    for chunk in chunks:
        text = (chunk.text or "").strip()
        meta = chunk.meta

        current_headings = [h for h in (getattr(meta, "headings", []) or []) if h]
        if current_headings:
            last_known_headings = current_headings

        effective_headings = current_headings if current_headings else last_known_headings

        doc_items = getattr(meta, "doc_items", []) or []
        is_table = any(getattr(item, "label", None) == "table" for item in doc_items)

        page_no = None
        captions = [c for c in (getattr(meta, "captions", []) or []) if c]

        if doc_items:
            origin = getattr(doc_items[0], "prov", None)
            if origin and len(origin) > 0:
                page_no = getattr(origin[0], "page_no", None)


        # Table chunk Handling
        if is_table:
            table_items = [item for item in doc_items if getattr(item, "label", None) == "table"]
            if table_items:
                table_item = table_items[0]

                try:
                    df = table_item.export_to_dataframe()
                    df.columns = [str(c).strip() for c in df.columns]

                    context_prefix = " > ".join(effective_headings) if effective_headings else "Table"
                    lines = [f"[{context_prefix}]"]

                    if captions:
                        lines.append(f"Table caption: {' | '.join(captions)}")

                    for _, row in df.iterrows():
                        row_facts = []
                        for col in df.columns:
                            val = _clean_cell_value(row[col])
                            if val:
                                row_facts.append(f"{col}: {val}")

                        if row_facts:
                            lines.append("- " + " | ".join(row_facts))

                    table_text = "\n".join(lines).strip()

                    if len(lines) <= 1:
                        # Fallback if dataframe extraction is empty
                        md = table_item.export_to_markdown().strip()
                        if effective_headings:
                            table_text = f"[{context_prefix}]\n{md}"
                        else:
                            table_text = md

                    text = table_text

                except Exception as e:
                    print(f"[warning] DataFrame export failed, falling back to markdown. Error: {e}")
                    md = table_item.export_to_markdown().strip()
                    context_prefix = " > ".join(effective_headings) if effective_headings else "Table"
                    text = f"[{context_prefix}]\n{md}" if md else ""

            if not text.strip():
                continue

            raw_parsed.append({
                "text": text,
                "page": page_no,
                "headings": effective_headings,
                "captions": captions,
                "is_table": True,
                "type": "table",
            })
            continue

        if len(text) < 15:
            continue

        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count < 5:
            continue

        raw_parsed.append({
            "text": text,
            "page": page_no,
            "headings": effective_headings,
            "captions": captions,
            "is_table": False,
            "type": "text",
        })


    # Stich Fragmented Text Chunks
    stitched_parsed = []
    for item in raw_parsed:
        if not stitched_parsed:
            stitched_parsed.append(item)
            continue

        prev = stitched_parsed[-1]

        if (
            item["type"] == "text"
            and prev["type"] == "text"
            and item["headings"] == prev["headings"]
            and item["page"] == prev["page"]
        ):
            prev["text"] += f"\n{item['text']}"
        else:
            stitched_parsed.append(item)


    # Context Injection
    final_parsed = []
    for item in stitched_parsed:
        if item["type"] == "text" and item["headings"] and not item["text"].startswith("["):
            context_header = " > ".join(item["headings"])
            item["text"] = f"[{context_header}]\n{item['text']}"

        final_parsed.append(item)


    # image_chunks
    for element, _level in doc.iterate_items():
        if not isinstance(element, PictureItem):
            continue

        page_no = None
        if getattr(element, "prov", None):
            page_no = getattr(element.prov[0], "page_no", None)

        pic_captions = []
        if getattr(element, "captions", None):
            for c in element.captions:
                if hasattr(c, "text"):
                    cap = (c.text or "").strip()
                else:
                    cap = str(c).strip()

                if cap:
                    pic_captions.append(cap)

        try:
            pil_image = element.get_image(doc)
        except Exception:
            pil_image = None

        if pil_image is None:
            continue

        final_parsed.append({
            "text": "",
            "page": page_no,
            "headings": [],
            "captions": pic_captions,
            "is_table": False,
            "type": "image",
            "image": pil_image,
        })

    return final_parsed
