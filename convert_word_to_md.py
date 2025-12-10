"""
Convert Word document to Markdown with tables and images.
"""
from docx import Document
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from pathlib import Path
import os
import re
from io import BytesIO

def extract_images_from_docx(docx_path, output_dir):
    """
    Extract all images from a Word document.

    Returns:
        dict: Mapping of image IDs to saved filenames
    """
    import zipfile
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    image_map = {}
    image_counter = 1

    # Open the docx as a zip file
    with zipfile.ZipFile(docx_path, 'r') as docx_zip:
        # List all files in the media folder
        media_files = [f for f in docx_zip.namelist() if f.startswith('word/media/')]

        for media_file in media_files:
            # Extract the image
            image_data = docx_zip.read(media_file)

            # Determine file extension
            ext = Path(media_file).suffix
            if not ext:
                # Try to detect from bytes
                try:
                    img = Image.open(BytesIO(image_data))
                    ext = '.' + img.format.lower()
                except:
                    ext = '.png'

            # Save with sequential naming
            filename = f"figure_{image_counter}{ext}"
            output_path = output_dir / filename

            with open(output_path, 'wb') as f:
                f.write(image_data)

            # Map the original filename to the new one
            image_map[media_file] = filename
            image_counter += 1

    return image_map

def get_paragraph_style(paragraph):
    """Determine the Markdown style for a paragraph."""
    style_name = paragraph.style.name.lower() if paragraph.style else ''

    if 'heading 1' in style_name:
        return '#'
    elif 'heading 2' in style_name:
        return '##'
    elif 'heading 3' in style_name:
        return '###'
    elif 'heading 4' in style_name:
        return '####'
    elif 'heading 5' in style_name:
        return '#####'
    elif 'heading 6' in style_name:
        return '######'
    elif 'title' in style_name:
        return '#'
    else:
        return None

def format_text_runs(paragraph):
    """Format text with bold, italic, etc."""
    text_parts = []

    for run in paragraph.runs:
        text = run.text
        if not text:
            continue

        # Apply formatting
        if run.bold and run.italic:
            text = f"***{text}***"
        elif run.bold:
            text = f"**{text}**"
        elif run.italic:
            text = f"*{text}*"

        text_parts.append(text)

    return ''.join(text_parts)

def table_to_markdown(table):
    """Convert a Word table to Markdown format."""
    if not table.rows:
        return ""

    # Extract table data
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            # Get cell text (combine all paragraphs)
            cell_text = ' '.join(p.text.strip() for p in cell.paragraphs if p.text.strip())
            cells.append(cell_text)
        rows.append(cells)

    if not rows:
        return ""

    # Determine column count
    max_cols = max(len(row) for row in rows)

    # Ensure all rows have the same number of columns
    for row in rows:
        while len(row) < max_cols:
            row.append('')

    # Build Markdown table
    md_lines = []

    # Header row
    header = rows[0]
    md_lines.append('| ' + ' | '.join(header) + ' |')

    # Separator
    md_lines.append('|' + '|'.join(['---' for _ in range(max_cols)]) + '|')

    # Data rows
    for row in rows[1:]:
        md_lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(md_lines)

def iter_block_items(parent):
    """
    Iterate through paragraphs and tables in document order.

    Yields:
        Paragraph or Table objects
    """
    from docx.document import Document
    if isinstance(parent, Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def convert_word_to_markdown(docx_path, output_md_path, images_dir='images'):
    """
    Convert Word document to Markdown.

    Args:
        docx_path: Path to .docx file
        output_md_path: Path for output .md file
        images_dir: Directory name for extracted images
    """
    docx_path = Path(docx_path)
    output_md_path = Path(output_md_path)
    images_dir = output_md_path.parent / images_dir

    print(f"Converting {docx_path} to {output_md_path}...")
    print(f"Images will be saved to: {images_dir}")

    # Extract images
    print("\nExtracting images...")
    try:
        image_map = extract_images_from_docx(docx_path, images_dir)
        print(f"  [OK] Extracted {len(image_map)} images")
    except Exception as e:
        print(f"  [WARNING] Image extraction warning: {e}")
        image_map = {}

    # Read document
    doc = Document(docx_path)

    # Track image counter for references
    image_counter = 1

    # Build Markdown content
    md_content = []

    print("\nConverting content...")
    table_count = 0
    paragraph_count = 0

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            paragraph_count += 1

            # Check for images in paragraph
            if block._element.xpath('.//pic:pic'):
                # This paragraph contains an image
                img_filename = f"figure_{image_counter}.png"
                if images_dir.exists() and (images_dir / img_filename).exists():
                    md_content.append(f"\n![Figure {image_counter}]({images_dir.name}/{img_filename})\n")
                    image_counter += 1
                continue

            # Get heading style
            heading_level = get_paragraph_style(block)

            # Format text
            text = format_text_runs(block)

            if not text.strip():
                continue

            if heading_level:
                md_content.append(f"\n{heading_level} {text}\n")
            else:
                md_content.append(f"{text}\n")

        elif isinstance(block, Table):
            table_count += 1
            md_table = table_to_markdown(block)
            if md_table:
                md_content.append(f"\n{md_table}\n")

    print(f"  [OK] Processed {paragraph_count} paragraphs")
    print(f"  [OK] Converted {table_count} tables")

    # Write to file
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))

    print(f"\n[OK] Conversion complete!")
    print(f"  Output: {output_md_path}")
    print(f"  Images: {images_dir}/")

    return output_md_path

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_word_to_md.py input.docx [output.md]")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = Path(input_file).stem + '.md'

    convert_word_to_markdown(input_file, output_file)
