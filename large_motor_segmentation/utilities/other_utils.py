def _pipline_merge_pdf(is_file_found=True):
    import PyPDF2
    import os

    merger = PyPDF2.PdfMerger()
    #
    # print(os.listdir(r'C:\Users\Lenovo\OneDrive\TM4\print'))
    # print(os.listdir(r'C:\Users\Lenovo\OneDrive\TM4\print').sort())
    files = ['0_Organisatorisches.pdf',
             '1_Kinematik eines starren Koerpers.pdf',
             '2_Kinetik eines starren Koerpers.pdf',
             '3_Eulerschen Gleichungen_1.pdf',
             '4_Eulerschen Gleichungen_2.pdf',
             '5_Bewegung von Starrkoerpersystemen.pdf',
             '6_Analytische Prinzipien der Mechanik.pdf',
             '7_Lagrange Gleichungen 2. Art.pdf',
             '8_Lagrange Gleichungen 2. Art (2).pdf',
             '9_Einfuehrung in die elementare Schwingungslehre.pdf',
             '10 Einfuehrung in die elementare Schwingungslehre (2).pdf',
             '11 Einfuehrung in die elementare Schwingungslehre (3).pdf',
             '12 Einfuehrung in die elementare Schwingungslehre (4).pdf',
             '13 Schwingungen in Systemen mit mehreren Freiheitsgraden.pdf',
             '14 Schwingungen in kontinuierlichen Systemen.pdf',
             'EM4_N-DOF-Oscillators.pdf']
    for filename in files:
        merger.append(PyPDF2.PdfReader(os.path.join(r'C:\Users\Lenovo\OneDrive\TM4\print', filename)))
    merger.write(os.path.join(r'C:\Users\Lenovo\OneDrive\TM4\print', 'TM4_merged.pdf'))

if __name__ == "__main__":
    _pipline_merge_pdf()