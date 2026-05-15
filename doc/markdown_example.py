from markitdown import MarkItDown
import fitz

FILE_PATH_1 = "./2025年学业指南.docx"
FILE_PATH_2 = "./2025学生手册.pdf"

md = MarkItDown()
result_1 = md.convert(FILE_PATH_1)
result_2 = ""

with fitz.open(FILE_PATH_2) as doc:
    for page in doc:
        result_2 += page.get_text() + "\n"

with open("./study_compass.md","w",encoding="utf-8") as f:
    f.write(result_1.text_content)

with open("./student_book.md","w",encoding="utf-8") as f:
    f.write(result_2)