"""
Convert CVD consolidated.docx to markdown format
"""
from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

def convert_docx_to_markdown(docx_path, md_path):
    """Convert a Word document to markdown format"""
    doc = Document(docx_path)

    markdown_content = []

    # Process each element in the document
    for element in doc.element.body:
        if isinstance(element, CT_P):
            para = Paragraph(element, doc)
            text = para.text.strip()

            if not text:
                markdown_content.append("")
                continue

            # Handle headings
            if para.style.name.startswith('Heading'):
                level = para.style.name.replace('Heading ', '')
                try:
                    level = int(level)
                    markdown_content.append(f"{'#' * level} {text}\n")
                except ValueError:
                    markdown_content.append(f"## {text}\n")
            else:
                # Regular paragraph
                # Handle bold and italic (basic formatting)
                formatted_text = text
                for run in para.runs:
                    if run.bold and run.italic:
                        formatted_text = formatted_text.replace(run.text, f"***{run.text}***")
                    elif run.bold:
                        formatted_text = formatted_text.replace(run.text, f"**{run.text}**")
                    elif run.italic:
                        formatted_text = formatted_text.replace(run.text, f"*{run.text}*")

                markdown_content.append(formatted_text + "\n")

        elif isinstance(element, CT_Tbl):
            table = Table(element, doc)
            markdown_content.append("\n")

            # Convert table to markdown
            for i, row in enumerate(table.rows):
                cells = [cell.text.strip() for cell in row.cells]
                markdown_content.append("| " + " | ".join(cells) + " |")

                # Add header separator after first row
                if i == 0:
                    markdown_content.append("| " + " | ".join(["---"] * len(cells)) + " |")

            markdown_content.append("\n")

    # Write to markdown file
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_content))

    print(f"Successfully converted {docx_path} to {md_path}")

if __name__ == "__main__":
    docx_file = "CVD consolidated.docx"
    md_file = "CVD_consolidated.md"

    try:
        convert_docx_to_markdown(docx_file, md_file)
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you get a 'No module named docx' error, install it with:")
        print("pip install python-docx")
