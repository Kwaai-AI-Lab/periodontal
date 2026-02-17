# Overleaf Setup Instructions for AD_FullText_v3

## File Organization in Overleaf

To properly compile the LaTeX document in Overleaf, organize your project files as follows:

### Project Structure

```
AD_FullText_v3/              (Root directory of your Overleaf project)
├── AD_FullText_v3.tex       (Main LaTeX file)
└── images/                  (Folder for all figures)
    ├── figure_1.png
    ├── figure_2.png
    ├── figure_3.png
    └── figure_4.png
```

## Step-by-Step Setup

### 1. Create a New Project in Overleaf
   - Go to Overleaf (https://www.overleaf.com)
   - Click "New Project" → "Blank Project"
   - Name it "AD_FullText_v3" or your preferred name

### 2. Upload the Main LaTeX File
   - Click the "Upload" button in the top-left corner of the project
   - Select `AD_FullText_v3.tex` from your local `AD_Model_v3` folder
   - The file will appear in the root directory

### 3. Create the Images Folder
   - In Overleaf, click the folder icon next to "New File"
   - Name the folder: `images`
   - Press Enter to create the folder

### 4. Upload Figure Images
   - Click on the `images` folder you just created
   - Click "Upload" and select all four image files:
     - `figure_1.png`
     - `figure_2.png`
     - `figure_3.png`
     - `figure_4.png`
   - All images should now be inside the `images/` folder

### 5. Verify File Paths
   The LaTeX file uses the following image paths:
   ```latex
   \includegraphics[width=0.8\textwidth]{images/figure_1.png}
   \includegraphics[width=0.8\textwidth]{images/figure_2.png}
   \includegraphics[width=0.8\textwidth]{images/figure_3.png}
   \includegraphics[width=0.8\textwidth]{images/figure_4.png}
   ```

   These paths are **relative to the main .tex file**, so the structure must match exactly.

### 6. Compile the Document
   - Click the green "Recompile" button
   - The PDF should compile successfully with all figures displayed
   - If you see errors about missing images, check that:
     - The `images` folder is at the root level (same level as .tex file)
     - All image files are inside the `images` folder
     - File names match exactly (case-sensitive)

## Troubleshooting

### Images Not Showing / File Not Found Errors

**Problem:** LaTeX can't find the image files

**Solutions:**
1. Check the folder structure matches the diagram above
2. Ensure the `images` folder is NOT nested inside another folder
3. Verify image file names are exactly: `figure_1.png`, `figure_2.png`, etc.
4. Check that files were uploaded to the correct folder (click on `images` folder to see its contents)

### Compilation Errors

**Problem:** Document won't compile

**Solutions:**
1. Check the log output for specific error messages
2. Ensure all required LaTeX packages are available (they should be by default in Overleaf)
3. Try compiling again - sometimes the first compilation fails on references

### Missing Fonts or Special Characters

**Problem:** Special characters (£, ε, etc.) not displaying correctly

**Solution:**
- Overleaf should handle these automatically
- If issues persist, ensure the compiler is set to **pdfLaTeX** (default)
- Check Settings → Compiler

## Alternative: Upload as ZIP

If you prefer to upload everything at once:

1. Create a folder on your computer named `AD_FullText_v3`
2. Copy `AD_FullText_v3.tex` into this folder
3. Copy the entire `images` folder into this folder
4. Compress the `AD_FullText_v3` folder as a .zip file
5. In Overleaf: New Project → Upload Project → Select your .zip file
6. Overleaf will automatically extract and organize the files

## Notes

- The LaTeX file includes all 51 references in BibTeX format
- Tables are formatted using the `booktabs` package for professional appearance
- All figures use `[htbp]` placement specifier (here, top, bottom, page)
- Document uses A4 paper size with 2.5cm margins
- Line spacing is set to 1.5 (one-and-a-half spacing)

## Compiling to PDF Locally

If you want to compile the document locally (outside Overleaf):

```bash
# Compile with pdflatex (run twice for references)
pdflatex AD_FullText_v3.tex
pdflatex AD_FullText_v3.tex
```

Ensure you have the following packages installed:
- booktabs
- graphicx
- natbib
- hyperref
- siunitx
- caption
- subcaption
