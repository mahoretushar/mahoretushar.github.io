from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# --- Page margins ---
for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.15)
    section.right_margin  = Inches(1.15)

# --- Default style ---
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)

HEADER_COLOR = RGBColor(0xE2, 0xEF, 0xD9)   # light-green table header (matches UGC format)
SECTION_SIZE = Pt(12)
BODY_SIZE    = Pt(11)


def para(text='', bold=False, size=None, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=4):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    if text:
        r = p.add_run(text)
        r.bold = bold
        r.font.name = 'Times New Roman'
        r.font.size = size or BODY_SIZE
    return p


def heading(text):
    p = para(text, bold=True, size=SECTION_SIZE, space_before=6, space_after=2)
    # Bottom border
    pPr = p.runs[0].element.getparent().get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '000000')
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p


def table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    # Header row
    hdr = t.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ''
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = BODY_SIZE
        # background
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'),   'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'),  'E2EFD9')
        tcPr.append(shd)
    # Data rows
    for ri, row_data in enumerate(rows):
        row = t.rows[ri + 1]
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = ''
            run = cell.paragraphs[0].add_run(val)
            run.font.name = 'Times New Roman'
            run.font.size = BODY_SIZE
    # Column widths
    if col_widths:
        for ci, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[ci].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def bullet(text):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = BODY_SIZE
    return p


def subbullet(text):
    p = doc.add_paragraph(style='List Bullet 2')
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = BODY_SIZE
    return p


# =====================================================================
# TITLE
# =====================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(4)
r = title.add_run('RESUME')
r.bold = True
r.underline = True
r.font.name = 'Times New Roman'
r.font.size = Pt(14)

# Name & contact
name_p = doc.add_paragraph()
name_p.paragraph_format.space_after = Pt(0)
rn = name_p.add_run('Tushar Ravindra Mahore')
rn.bold = True
rn.font.name = 'Times New Roman'
rn.font.size = Pt(12)

para('Ph.D. Researcher (SIT, Pune) | M.Tech (Computer Science and Engineering), 2018', size=BODY_SIZE, space_after=0)

mob = doc.add_paragraph()
mob.paragraph_format.space_after = Pt(0)
r1 = mob.add_run('Mobile: ')
r1.bold = True; r1.font.name = 'Times New Roman'; r1.font.size = BODY_SIZE
r2 = mob.add_run('+91 7588085340')
r2.font.name = 'Times New Roman'; r2.font.size = BODY_SIZE

eml = doc.add_paragraph()
eml.paragraph_format.space_after = Pt(6)
r1 = eml.add_run('Email: ')
r1.bold = True; r1.font.name = 'Times New Roman'; r1.font.size = BODY_SIZE
r2 = eml.add_run('mahoretushar@gmail.com')
r2.font.name = 'Times New Roman'; r2.font.size = BODY_SIZE

# Horizontal rule via bottom border on empty paragraph
hr = doc.add_paragraph()
hr.paragraph_format.space_after = Pt(6)
pPr = hr._p.get_or_add_pPr()
pBdr = OxmlElement('w:pBdr')
bot = OxmlElement('w:bottom')
bot.set(qn('w:val'), 'single'); bot.set(qn('w:sz'), '12')
bot.set(qn('w:space'), '1');    bot.set(qn('w:color'), '000000')
pBdr.append(bot); pPr.append(pBdr)


# =====================================================================
# PROFESSIONAL SUMMARY
# =====================================================================
heading('Professional Summary')
summary = (
    "Accomplished educator and researcher with 7+ years of teaching experience across six institutions, "
    "combined with active research in Natural Language Processing and Artificial Intelligence. "
    "Author of 15+ publications in reputed IEEE, Springer, Elsevier, AIP, and IOP venues; "
    "patent holder and copyright registrant. Experienced in curriculum development, student assessment, "
    "academic administration, and data-driven instruction. "
    "Proven ability to create engaging and inclusive learning experiences utilizing diverse instructional "
    "methodologies and technology tools. Adept at fostering academic growth and cultivating a love for "
    "learning while maintaining a strong research orientation. Excellent communication and interpersonal "
    "skills with a demonstrated ability to build positive relationships with students, colleagues, and "
    "stakeholders. Committed to contributing meaningfully to academic and research-oriented environments "
    "through teaching excellence, scholarly output, and institutional development."
)
para(summary, space_after=6)


# =====================================================================
# TEACHING EXPERIENCE
# =====================================================================
heading('Teaching Experience')
table(
    headers=['Designation', 'Institute', 'Start Date', 'End Date'],
    rows=[
        ['Assistant Professor', 'Indira College of Engineering and Management (ICEM), Pune', 'Feb 2026', 'Present'],
        ['Assistant Professor', 'Pimpri Chinchwad University (PCU), Pune',                  'July 2024', 'Feb 2026'],
        ['Instructor',          'Great Learning',                                            'March 2023', 'Aug 2025'],
        ['Assistant Professor', 'Sipna College of Engineering and Technology (SCOET), Amravati', 'July 2022', 'June 2024'],
        ['Assistant Professor', 'Dr. Rajendra Gode Institute of Technology & Research (DRGITR), Amravati', '2019', '2022'],
        ['Assistant Professor\n(CHB + M.Tech)', 'Government College of Engineering, Amravati (GCOEA)', '2018', '2019'],
    ],
    col_widths=[1.6, 3.0, 1.0, 1.0],
)


# =====================================================================
# QUALIFICATION DETAILS
# =====================================================================
heading('Qualification Details')
table(
    headers=['Qualification', 'University / Institute', 'Year', 'Specialization', 'Division'],
    rows=[
        ['Ph.D. (Pursuing)', 'Symbiosis Institute of Technology (SIT), Pune', '2023 – Present', 'Computer Science & Engineering', '—'],
        ['M.Tech', 'GCOEA, Amravati', '2018', 'Computer Science & Engineering', 'Distinction'],
        ['B.E.',   'SGBAU',           '2014', 'Computer Science & Engineering', 'First Class'],
        ['HSC',    'Amravati Board',  '2010', 'General Science',                 'Second Class'],
        ['SSC',    'Amravati Board',  '2006', 'English',                          'Second Class'],
    ],
    col_widths=[1.2, 2.2, 1.2, 2.0, 1.0],
)


# =====================================================================
# SUBJECTS TAUGHT
# =====================================================================
heading('Subjects Taught')
table(
    headers=['Subject Area', 'Subjects Taught'],
    rows=[
        ['AI & Machine Learning',        'Artificial Neural Networks'],
        ['Data Science & Analytics',     'Data Science & Statistics, Big Data Analytics, Data Mining, Data Modeling & Visualization'],
        ['Database Systems',             'Database Management Systems, Advanced Databases, Database Systems'],
        ['Programming & Algorithms',     'Python Programming, JAVA, Design and Analysis of Algorithms, Computer Programming (C-Language)'],
        ['Networking',                   'Data Communication & Computer Networks'],
        ['Theoretical Computer Science', 'Theory of Computation, Discrete Structure and Graph Theory'],
        ['Other',                        'Dev-Ops (Practical), Operations Research and Management, Network Security'],
    ],
    col_widths=[2.2, 5.4],
)


# =====================================================================
# SKILLS
# =====================================================================
heading('Skills')
table(
    headers=['Skill Type', 'Skills'],
    rows=[
        ['Subject Matter Expertise',      'NLP, AI/ML, Data Science (Python/R), Cyber Security, Data Structures'],
        ['NLP & AI Research',             'Transformers, Abstractive Summarization, RAG, LangChain, Edge AI, Crisis Informatics'],
        ['Programming Languages',         'Python, C, Mojo, Perl, Ruby, Julia, Rust, Lua'],
        ['Data Science & Analytics Tools','KNIME, Power BI, Tableau, Anaconda, BigQuery, Databricks, Pandas, Scikit-learn'],
        ['Database Systems',              'Oracle, MySQL, MongoDB, PostgreSQL'],
        ['Operating Systems',             'RedHat, Ubuntu, Fedora, Kali Linux, MacOS, Windows'],
        ['Other Technical Skills',        'Terminal Commands, Shell Scripting, Vim, Emacs, Git, GitHub, Docker'],
        ['Teaching & Mentoring',          'Ability to recognize and respond to diverse student needs, identify and address individual learning gaps, effective classroom management and organization, develop and implement engaging lesson plans and assessments, strong communication and interpersonal skills to foster a positive learning environment.'],
    ],
    col_widths=[2.2, 5.4],
)


# =====================================================================
# WORKSHOP / TRAINING / EVENTS CONDUCTED
# =====================================================================
heading('Workshop/Training/Events Conducted')
table(
    headers=['Workshop/Training Title', 'Duration'],
    rows=[
        ['IDEATHON 01 (College Level Hackathon)',                                '1 day'],
        ['IDEATHON 2.0 (24 Hrs International Hackathon)',                        '2 days'],
        ['Data Science & Statistics',                                            '3 months'],
        ['Data Analytics',                                                       '3 months'],
        ['Python',                                                               '1 month'],
        ['5 Days Training on Python for Campus Placements',                      '5 days'],
        ['5 Days Workshop on "Programming with PYTHON"',                         '5 days'],
        ['3 Days Workshop on "C Programming"',                                   '3 days'],
        ['4 Days Workshop on "Data Structure"',                                  '4 days'],
        ['8 Days Workshop on "Machine Learning and Data Science with R"',        '8 days'],
    ],
    col_widths=[5.6, 2.0],
)


# =====================================================================
# CERTIFICATIONS
# =====================================================================
heading('Certifications')
table(
    headers=['Certification Title', 'Provider', 'Date Earned'],
    rows=[
        ['Natural Language Processing Specialization (4 Courses)',                  'DeepLearning.AI — Coursera',                                          'July 2025'],
        ['Generative AI Engineering with LLMs Specialization',                      'IBM — Coursera',                                                      'July 2025'],
        ['Red Hat System Administration I (RH124 – RHCSA)',                         'Red Hat',                                                             'October 2025'],
        ['One Week FDP: AI for Sustainable Development',                            'VIT Pune — IEEE Pune Section',                                        'May 2026'],
        ['One Week FDP: Integrating ML and AI for Scalable IoT Solutions',          'VIT Pune — IEEE CTSoc, Pune Chapter',                                 'April 2026'],
        ['FDP on AI Tools — National Level, AICTE',                                 'Dayananda Sagar Academy of Technology & Management, Bengaluru',       'February 2025'],
        ['AI Fluency: Framework & Foundations',                                     'Anthropic',                                                           'April 2026'],
        ['ISTE STTP on Research Methodology & Use of ICT Tools',                    'P.R. Pote (Patil) College of Engineering & Management, Amravati',     'February 2023'],
        ['Introduction to Data Science',                                            'Cisco Networking Academy',                                            'May 2023'],
        ['AICTE STTP on Data Analytics in Machine Learning Techniques',             'Sri Venkateshwara College of Engineering, Bengaluru',                 'March 2021'],
        ['Certification in Python for Data Analysis, Data Science and ML with Pandas', 'MOOC',                                                            'April 2021'],
        ['Certification in Basics of Python',                                       'Infosys Springboard',                                                 'May 2022'],
        ['Certification in Linux Command Line Basics to Advance',                   'Udemy',                                                               'May 2022'],
    ],
    col_widths=[3.0, 2.6, 1.0],
)


# =====================================================================
# AWARDS / IPR
# =====================================================================
heading('Awards / IPR')
table(
    headers=['Title', 'Type', 'Date'],
    rows=[
        ['"An Analysis Based System for Identifying Media to Broadcast Positive Thoughts in Covid-19 Pandemic"', 'Copyright', 'March 8, 2021'],
        ['"AI Based Smart System For The Early Detection Of Omicron And Covid-19 Symptoms"',                     'Patent',    'June 2022'],
    ],
    col_widths=[4.0, 1.2, 1.4],
)


# =====================================================================
# RESPONSIBILITIES HANDLED
# =====================================================================
heading('Responsibilities Handled')
table(
    headers=['Responsibility', 'Duration', 'Institute'],
    rows=[
        ['Internal and External Examiner, SPPU APR–MAY 2026',  'May 2026',              'ICEM'],
        ['University Examination — Assistant Senior Supervisor','May 2025',              'PCU'],
        ['Academic Coordinator',                                'July 2025',             'PCU'],
        ['Departmental ERP Coordinator',                        'July 2024 – Feb 2026',  'PCU'],
        ['Departmental Result Analysis Incharge',               'July 2024 – Feb 2026',  'PCU'],
        ['Academic Audit Departmental Incharge',                'July 2022 – June 2024', 'SCOET'],
        ['Student Activity Member (Departmental)',              'July 2022 – June 2024', 'SCOET'],
        ['Member, Admissions Process',                          '2023–2024',             'SCOET'],
        ['Head of Department',                                  'March 2022 – June 2022','DRGITR'],
        ['Co-Incharge, E-Sc Centre (Admissions)',               '2021–2022',             'DRGITR'],
        ['Institute Level In-Charge of Website',                '2021–2022',             'DRGITR'],
        ['Co-Incharge, E-Sc Centre (Admissions)',               '2020–2021',             'DRGITR'],
        ['Departmental Incharge, SGBAU Summer Exam',            'Summer 2020',           'DRGITR'],
        ['FC Centre, Admissions Process',                       '2019–2020',             'DRGITR'],
        ['Exam Committee Member',                               'Winter 2019',           'DRGITR'],
        ['Departmental Incharge of Feedback',                   '2019–2020',             'DRGITR'],
        ['NAAC Criteria 3 Incharge',                            '2019–2020',             'DRGITR'],
        ['Member, Sports Department',                           '2019–2021',             'DRGITR'],
        ['Coordinator, M.E. Program',                           '2019–2021',             'DRGITR'],
        ['Departmental Incharge of Timetable',                  '2020–2021',             'DRGITR'],
    ],
    col_widths=[3.2, 2.0, 1.5],
)


# =====================================================================
# FDP's / WORKSHOPS ATTENDED
# =====================================================================
heading("FDP's / Workshops Attended")
fdps = [
    'One Week FDP on "AI for Sustainable Development", VIT Pune in association with IEEE Pune Section. May 4–8, 2026.',
    'One Week FDP on "Integrating Machine Learning and AI for Scalable IoT Solutions", VIT Pune — IEEE CTSoc, Pune Chapter. April 13–17, 2026.',
    'FDP on "AI Tools" — National Level, AICTE, Dayananda Sagar Academy of Technology & Management, Bengaluru. February 17–21, 2025.',
    'One Week Faculty Development Program on "Research Methodology and Use of ICT Tools", Sponsored by ISTE, organized by P.R. Pote (Patil) College of Engineering and Management, Amravati. February 20–25, 2023.',
    'Faculty Development Program on "Amazon Web Services", Dr. D.Y. Patil Institute of Engineering, Management & Research, Akurdi, Pune. August 22–27, 2022.',
    'National Level One Week FDP on "Multi-Technology", organized by DRGIT & R, Amravati in collaboration with Brainovision Solutions India Pvt. Ltd. and National Youth Council of India. June 28 – July 3, 2021.',
    'AICTE Sponsored One Week Online STTP on "Data Analytics in Machine Learning Techniques", Sri Venkateshwara College of Engineering, Bengaluru. March 22–27, 2021.',
    '5 Day National Level Online FDP on "Artificial Intelligence", DRGIT & R in association with National Youth Council of India and Brainvision Solutions Pvt. Ltd. May 22–26, 2020.',
    'One-week International Research Oriented Program, MM University / Elsevier / RAx Labs. October 26–31, 2020.',
    'Participated in "LogiTRIx, Advanced Autonomous Robotics Workshop" conducted by ThinksLABS SINE IIT-Bombay, 2011.',
    'Participated in "LoopHole — Ethical Hacking Workshop" held by Kyrion Digital Securities, 2012.',
    'Attended one-day workshop on Cyber Security organized by HVPM College of Engineering and Technology, 2016.',
    'Completed Diploma in Software Testing, Seed Infotech. July 14 – August 8, 2014.',
]
for f in fdps:
    bullet(f)

p_gap = doc.add_paragraph()
p_gap.paragraph_format.space_after = Pt(4)


# =====================================================================
# PUBLICATIONS & PRESENTATIONS
# =====================================================================
heading('Publications & Presentations')

pb = doc.add_paragraph()
pb.paragraph_format.space_after = Pt(2)
r1 = pb.add_run('Total Papers Published: ')
r1.bold = True; r1.font.name = 'Times New Roman'; r1.font.size = BODY_SIZE
r2 = pb.add_run('15+')
r2.font.name = 'Times New Roman'; r2.font.size = BODY_SIZE

# Conference Papers
cp = doc.add_paragraph()
cp.paragraph_format.space_after = Pt(2)
rc = cp.add_run('Conference Papers:')
rc.bold = True; rc.font.name = 'Times New Roman'; rc.font.size = BODY_SIZE

conf_papers = [
    '"Cyberbullying Classification Using Natural Language Processing and Machine Learning Techniques," IEEE ACROSET 2024, Acropolis Institute of Technology & Research, Indore, September 27–28, 2024.',
    '"A Survey on Credit Card Fraud Detection using Machine Learning and Deep Learning Techniques," AIP Conference Proceedings, Vol. 2800(1), 020118, 2023.',
    '"A Survey on Bigdata in Healthcare," International Conference on Data Analytics & Management: An Indo-European Conference (ICDAM-2021).',
    '"A survey on credit card fraud detection using various machine learning and deep learning techniques," Fourth Scientific Conference for Electrical Engineering Techniques Research (EETR2022).',
    '"A Survey in Various Attacks Possible in Authentication," IJARSE, Vol. 6, Issue 4, April 2017.',
    '"Safe and Secure Graphical Authentication System," IJSR, Vol. 6, Issue 4, April 2017.',
    '"Secure Graphical Password Scheme," IJRPET, Vol. 3, Issue 3, March 2017.',
    '"Security Challenges and Issues in IoT (Internet of Things)," IJSER, Vol. 7, Issue 2, February 2016.',
    '"A Survey on Various Authentication Techniques and Graphical Passwords," IJATES, Vol. 5, Issue 4, April 2017.',
    'Paper presented at 1st International Conference on Computational Research and Data Analytics (ICCRDA-2020).',
]
for cp_text in conf_papers:
    subbullet(cp_text)

# Book Chapters
bch = doc.add_paragraph()
bch.paragraph_format.space_after = Pt(2)
rb = bch.add_run('Book Chapters:')
rb.bold = True; rb.font.name = 'Times New Roman'; rb.font.size = BODY_SIZE

book_chapters = [
    '"An Explainable Hybrid TabTransformer–Random Forest Model for Biometric Security in IoMT Healthcare Systems," Academic Press — Recent Advances in Computational Intelligence Applications for Biometrics and Biomedical Devices, pp. 285–300, 2026.',
    '"Coordinated Response Strategies: Swarm Robotics for Crisis Management," Auerbach Publications — AI and Machine Learning for Mechanical and Electrical Engineering, pp. 182–197, 2025.',
    '"Detection of Multi-class Skin Cancer using Stochastic Gradient Descent Augmentation Model and Activation Mapping," Journal of Innovative Image Processing, Vol. 7(4), pp. 1415–1435, 2025.',
    '"Student Attendance Monitoring System Using Facial Recognition," Springer Nature Singapore — Innovative Computing and Communications (ICICC 2022), Vol. 3, pp. 613–620.',
    '"Food Classification Using Deep Learning Algorithm," Springer Nature Singapore — Innovative Computing and Communications (ICICC 2022), Vol. 3, pp. 717–724.',
    '"Credit Card Fraud Detection Using Machine Learning and Deep Learning Approaches," Springer Nature Singapore — Innovative Computing and Communications (ICICC 2022), Vol. 3, pp. 621–628.',
    '"Comparative Analysis of Detection of Email Spam with Machine Learning Approaches," IOP Conference Series: Materials Science and Engineering, Vol. 1022(1), 012113, 2021.',
]
for bc_text in book_chapters:
    subbullet(bc_text)

# Journal Articles
ja = doc.add_paragraph()
ja.paragraph_format.space_after = Pt(2)
rj = ja.add_run('Journal Articles / Other:')
rj.bold = True; rj.font.name = 'Times New Roman'; rj.font.size = BODY_SIZE

articles = [
    '"Secure Graphical Password Scheme," JournalNX, Vol. 3(03), pp. 144–147, 2017.',
    '"Review Classification Approach for User Sentiment Analysis."',
    '"Result Analysis of User Review for Sentiment Classification."',
    '"Review on Security Threats of Cloud Environment," IJAIT, March 2022.',
]
for art in articles:
    subbullet(art)

p_gap2 = doc.add_paragraph()
p_gap2.paragraph_format.space_after = Pt(4)


# =====================================================================
# PROJECTS & SOFTWARE DEVELOPMENT
# =====================================================================
heading('Projects & Software Development')
bullet('Developed software for generating leaving certificates (2020–2021).')
bullet('Implemented a project for result analysis at an educational institute (2020–2021).')
bullet('Developed and implemented a website for an institute within a campus (2021–2022).')

p_gap3 = doc.add_paragraph()
p_gap3.paragraph_format.space_after = Pt(4)


# =====================================================================
# PERSONAL INFORMATION
# =====================================================================
heading('Personal Information')

personal = [
    ('DOB:',             '03rd May 1991'),
    ("Father's Name:",   'Ravindra Eknathrao Mahore'),
    ('Hobbies:',         'Gym, Reading, Sports'),
    ('Languages Known:', 'English, Hindi, Marathi'),
    ('Address:',         'Address One, Peninsula Land, Mamurdi'),
]
for label, value in personal:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    rl = p.add_run(label + ' ')
    rl.bold = True; rl.font.name = 'Times New Roman'; rl.font.size = BODY_SIZE
    rv = p.add_run(value)
    rv.font.name = 'Times New Roman'; rv.font.size = BODY_SIZE


# =====================================================================
# SAVE
# =====================================================================
out = '/Users/wolf/Documents/Everything/10_Projects/14_Personal/Website/Website/assets/pdf/TusharRMahore_Resume_2026.docx'
doc.save(out)
print(f'Saved: {out}')
