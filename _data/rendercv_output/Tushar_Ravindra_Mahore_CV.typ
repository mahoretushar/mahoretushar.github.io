// Import the rendercv function and all the refactored components
#import "@preview/rendercv:0.3.0": *

// Apply the rendercv template with custom configuration
#show: rendercv.with(
  name: "Tushar Ravindra Mahore",
  title: "Tushar Ravindra Mahore - CV",
  footer: context { [#emph[Tushar Ravindra Mahore -- #str(here().page())\/#str(counter(page).final().first())]] },
  top-note: [ #emph[Last updated in May 2026] ],
  locale-catalog-language: "en",
  text-direction: ltr,
  page-size: "a4",
  page-top-margin: 0.7in,
  page-bottom-margin: 0.7in,
  page-left-margin: 0.7in,
  page-right-margin: 0.7in,
  page-show-footer: false,
  page-show-top-note: false,
  colors-body: rgb(0, 0, 0),
  colors-name: rgb(0, 79, 144),
  colors-headline: rgb(0, 79, 144),
  colors-connections: rgb(0, 79, 144),
  colors-section-titles: rgb(0, 79, 144),
  colors-links: rgb(0, 79, 144),
  colors-footer: rgb(128, 128, 128),
  colors-top-note: rgb(128, 128, 128),
  typography-line-spacing: 0.6em,
  typography-alignment: "justified",
  typography-date-and-location-column-alignment: right,
  typography-font-family-body: "Source Sans 3",
  typography-font-family-name: "Source Sans 3",
  typography-font-family-headline: "Source Sans 3",
  typography-font-family-connections: "Source Sans 3",
  typography-font-family-section-titles: "Source Sans 3",
  typography-font-size-body: 10pt,
  typography-font-size-name: 30pt,
  typography-font-size-headline: 10pt,
  typography-font-size-connections: 10pt,
  typography-font-size-section-titles: 1.4em,
  typography-small-caps-name: false,
  typography-small-caps-headline: false,
  typography-small-caps-connections: false,
  typography-small-caps-section-titles: false,
  typography-bold-name: true,
  typography-bold-headline: false,
  typography-bold-connections: false,
  typography-bold-section-titles: true,
  links-underline: false,
  links-show-external-link-icon: false,
  header-alignment: center,
  header-photo-width: 3.5cm,
  header-space-below-name: 0.7cm,
  header-space-below-headline: 0.7cm,
  header-space-below-connections: 0.7cm,
  header-connections-hyperlink: true,
  header-connections-show-icons: true,
  header-connections-display-urls-instead-of-usernames: false,
  header-connections-separator: "",
  header-connections-space-between-connections: 0.5cm,
  section-titles-type: "with_partial_line",
  section-titles-line-thickness: 0.5pt,
  section-titles-space-above: 0.5cm,
  section-titles-space-below: 0.3cm,
  sections-allow-page-break: true,
  sections-space-between-text-based-entries: 0.3em,
  sections-space-between-regular-entries: 1.2em,
  entries-date-and-location-width: 4.15cm,
  entries-side-space: 0.2cm,
  entries-space-between-columns: 0.1cm,
  entries-allow-page-break: false,
  entries-short-second-row: true,
  entries-degree-width: 1cm,
  entries-summary-space-left: 0cm,
  entries-summary-space-above: 0cm,
  entries-highlights-bullet:  "•" ,
  entries-highlights-nested-bullet:  "•" ,
  entries-highlights-space-left: 0.15cm,
  entries-highlights-space-above: 0cm,
  entries-highlights-space-between-items: 0cm,
  entries-highlights-space-between-bullet-and-text: 0.5em,
  date: datetime(
    year: 2026,
    month: 5,
    day: 18,
  ),
)


= Tushar Ravindra Mahore

#connections(
  [#link("mailto:mahoretushar@gmail.com", icon: false, if-underline: false, if-color: false)[#connection-with-icon("envelope")[mahoretushar\@gmail.com]]],
  [#link("tel:+91-75880-85340", icon: false, if-underline: false, if-color: false)[#connection-with-icon("phone")[075880 85340]]],
  [#connection-with-icon("location-dot")[Pune, Maharashtra, India]],
  [#link("https://mahoretushar.github.io/", icon: false, if-underline: false, if-color: false)[#connection-with-icon("link")[mahoretushar.github.io]]],
  [#link("https://github.com/mahoretushar", icon: false, if-underline: false, if-color: false)[#connection-with-icon("github")[mahoretushar]]],
  [#link("https://scholar.google.com/citations?user=nETPgMwAAAAJ", icon: false, if-underline: false, if-color: false)[#connection-with-icon("graduation-cap")[Google Scholar]]],
  [#link("https://orcid.org/0009-0000-9406-1178", icon: false, if-underline: false, if-color: false)[#connection-with-icon("orcid")[0009-0000-9406-1178]]],
)


== Summary

- Assistant Professor, Department of AI & Data Science, Indira College of Engineering and Management (ICEM), Pune. PhD researcher at Symbiosis Institute of Technology (SIT), Pune — building edge-deployable NLP pipelines for real-time disaster situation summarization. 7+ years of teaching experience across six institutions. 15+ publications in IEEE, Springer, Elsevier, AIP, and IOP venues. Patent holder and copyright registrant.


== Education

#education-entry(
  [
    #strong[Symbiosis Institute of Technology (SIT)], Computer Science & Engineering

    - Research: Edge-deployable NLP pipelines for real-time disaster situation summarization

    - Focus areas: Natural Language Processing, Crisis Informatics, Edge AI

  ],
  [
    Lavale, Pune, Maharashtra

    Jan 2023 – present

  ],
  degree-column: [
    
  ],
)

#education-entry(
  [
    #strong[Government College of Engineering, Amravati (GCOEA)], Computer Science & Engineering

    - Distinction — top performance in program

    - Thesis: Secure Graphical Password Scheme (published in JournalNX, 2017)

  ],
  [
    Amravati, Maharashtra

    Jan 2016 – Jan 2018

  ],
  degree-column: [
    
  ],
)

#education-entry(
  [
    #strong[Sant Gadge Baba Amravati University (SGBAU)], Computer Science & Engineering

    - First Class

  ],
  [
    Amravati, Maharashtra

    Jan 2010 – Jan 2014

  ],
  degree-column: [
    
  ],
)

== Experience

#regular-entry(
  [
    #strong[Indira College of Engineering and Management (ICEM)], Assistant Professor

    - Department of Artificial Intelligence and Data Science, SPPU affiliated

    - Teaching: Artificial Neural Networks (317531), T.E. AI & DS (2019 pattern)

    - Internal and External Examiner, SPPU APR–MAY 2026 examinations

  ],
  [
    Pune, Maharashtra

    Feb 2026 – present

    

    4 months

  ],
)

#regular-entry(
  [
    #strong[Pimpri Chinchwad University (PCU)], Assistant Professor

    - Teaching: Data Science & Analytics; Data Communication & Computer Networks; Java Programming; Data Modeling & Visualization

    - Academic Coordinator (July 2025)

    - University Examination — Assistant Senior Supervisor (May 2025)

    - Departmental ERP Coordinator and Result Analysis Incharge (July 2024)

  ],
  [
    Pune, Maharashtra

    July 2024 – Feb 2026

    

    1 year 8 months

  ],
)

#regular-entry(
  [
    #strong[Great Learning], Instructor

    - Delivered online courses in Data Science, Analytics, and Python

  ],
  [
    Online

    Mar 2023 – Aug 2025

    

    2 years 6 months

  ],
)

#regular-entry(
  [
    #strong[Sipna College of Engineering and Technology (SCOET)], Assistant Professor

    - Academic Audit Departmental Incharge (July 2022 – June 2024)

    - Member, Admissions Process (2023–2024)

  ],
  [
    Amravati, Maharashtra

    July 2022 – June 2024

    

    2 years

  ],
)

#regular-entry(
  [
    #strong[Dr. Rajendra Gode Institute of Technology & Research (DRGITR)], Assistant Professor

    - Head of Department (March 2022 – June 2022)

    - Institute Level In-Charge of Website (2021–2022)

    - Coordinator, M.E. Program (2019–2021)

    - NAAC Criteria 3 Incharge (2019–2020)

  ],
  [
    Amravati, Maharashtra

    Jan 2019 – Jan 2022

    

    3 years 1 month

  ],
)

#regular-entry(
  [
    #strong[Government College of Engineering, Amravati (GCOEA)], Assistant Professor (CHB + M.Tech)

    - Taught while completing M.Tech program

  ],
  [
    Amravati, Maharashtra

    Jan 2018 – Jan 2019

    

    1 year 1 month

  ],
)

== Courses Taught

#strong[ICEM, Pune (2026–Present):] Artificial Neural Networks (317531) — T.E. AI & DS, SPPU 2019 Pattern

#strong[PCU, Pune (2024–2026):] Data Science & Analytics · Data Communication & Computer Networks · Java Programming · Data Modeling & Visualization

== Publications

#regular-entry(
  [
    #strong[An Explainable Hybrid TabTransformer–Random Forest Model for Biometric Security in IoMT Healthcare Systems]

    Sagar Dhanraj Pande, #strong[Tushar Ravindra Mahore], Anushka Ashish Joshi, Vivek B. Kute, Sumit S. Sagne, Ankush Vasant Dahat

    (Academic Press — Recent Advances in Computational Intelligence Applications for Biometrics and Biomedical Devices, pp. 285–300)

  ],
  [
    Jan 2026

  ],
)

#regular-entry(
  [
    #strong[Detection of Multi-class Skin Cancer using Stochastic Gradient Descent Augmentation Model and Activation Mapping]

    Ankush Vasant Dahat, #strong[Tushar Ravindra Mahore], Anushka Ashish Joshi, Sagar Dhanraj Pande

    (Journal of Innovative Image Processing, Vol. 7(4), pp. 1415–1435)

  ],
  [
    Jan 2025

  ],
)

#regular-entry(
  [
    #strong[Coordinated Response Strategies: Swarm Robotics for Crisis Management]

    Renu R. Dandge, #strong[Tushar R. Mahore], Ankush Vasant Dahat, Pallavi H. Dhole, Nikhilesh P. Mankar, Sagar Dhanraj Pande

    (Auerbach Publications — AI and Machine Learning for Mechanical and Electrical Engineering, pp. 182–197)

  ],
  [
    Jan 2025

  ],
)

#regular-entry(
  [
    #strong[Cyberbullying Classification Using Natural Language Processing and Machine Learning Techniques]

    Nikhilesh Pramod Mankar, #strong[Tushar Mahore], Anushka A. Joshi, Sagar Dhanraj Pande, Aditya Khamparia, Ankita Shrikrushna Nathe, Fadi Al Turjman

    (IEEE — 2024 International Conference on Advances in Computing Research on Science Engineering and Technology (ACROSET), pp. 1–5)

  ],
  [
    Jan 2024

  ],
)

#regular-entry(
  [
    #strong[A Survey on Credit Card Fraud Detection using Machine Learning and Deep Learning Techniques]

    Ashvini S. Gorte, S. W. Mohod, R. R. Keole, #strong[T. R. Mahore], Sagar Pande

    (AIP Conference Proceedings, Vol. 2800(1), 020118)

  ],
  [
    Jan 2023

  ],
)

#regular-entry(
  [
    #strong[Credit Card Fraud Detection Using Machine Learning and Deep Learning Approaches]

    Ashvini S. Gorte, S. W. Mohod, R. R. Keole, #strong[T. R. Mahore], Sagar Pande

    (Springer Nature Singapore — Innovative Computing and Communications (ICICC 2022), Vol. 3, pp. 621–628)

  ],
  [
    Jan 2022

  ],
)

#regular-entry(
  [
    #strong[Food Classification Using Deep Learning Algorithm]

    R. V. Jamnekar, R. R. Keole, S. W. Mohod, #strong[T. R. Mahore], Sagar Pande

    (Springer Nature Singapore — Innovative Computing and Communications (ICICC 2022), Vol. 3, pp. 717–724)

  ],
  [
    Jan 2022

  ],
)

#regular-entry(
  [
    #strong[Student Attendance Monitoring System Using Facial Recognition]

    Reshma B. Wankhade, S. W. Mohod, R. R. Keole, #strong[T. R. Mahore], Sagar Dhanraj Pande

    (Springer Nature Singapore — Innovative Computing and Communications (ICICC 2022), Vol. 3, pp. 613–620)

  ],
  [
    Jan 2022

  ],
)

#regular-entry(
  [
    #strong[A Survey on Big Data in Healthcare]

    Sheena J. Popli, #strong[Tushar R. Mahore], Nikhil E. Karale, Sagar Pande

    (International Conference on Innovative Computing and Communication (ICICC 2021))

  ],
  [
    Jan 2021

  ],
)

#regular-entry(
  [
    #strong[Comparative Analysis of Detection of Email Spam with Machine Learning Approaches]

    Mangena Venu Madhavan, Sagar Pande, Pooja Umekar, #strong[Tushar Mahore], Dhiraj Kalyankar

    (IOP Conference Series: Materials Science and Engineering, Vol. 1022(1), 012113)

  ],
  [
    Jan 2021

  ],
)

#regular-entry(
  [
    #strong[Secure Graphical Password Scheme]

    #strong[Tushar R. Mahore], A. V. Deorankar

    (JournalNX, Vol. 3(03), pp. 144–147)

  ],
  [
    Jan 2017

  ],
)

== IPR & Awards

#strong[Patent: AI Based Smart System For Early Detection Of Omicron And Covid-19 Symptoms:] Indian Patent Office, June 2022

#strong[Copyright: Analysis Based System for Identifying Media to Broadcast Positive Thoughts:] Copyright Office, India, March 2021

== Certifications

#strong[Natural Language Processing Specialization (4 Courses):] DeepLearning.AI — Coursera, July 2025

#strong[Generative AI Engineering with LLMs Specialization:] IBM — Coursera, July 2025

#strong[Red Hat System Administration I (RH124 – RHCSA):] Red Hat, October 2025

#strong[One Week FDP: AI for Sustainable Development:] VIT Pune — IEEE Pune Section, May 2026

#strong[One Week FDP: Integrating ML and AI for Scalable IoT Solutions:] VIT Pune — IEEE CTSoc, Pune Chapter, April 2026

#strong[FDP on AI Tools — National Level, AICTE:] Dayananda Sagar Academy of Technology & Management, Bengaluru, February 2025

#strong[AI Fluency: Framework & Foundations:] Anthropic, April 2026

#strong[ISTE STTP on Research Methodology & Use of ICT Tools:] P.R. Pote (Patil) College, Amravati, February 2023

#strong[Introduction to Data Science:] Cisco Networking Academy, May 2023

#strong[AICTE STTP on Data Analytics in Machine Learning Techniques:] Sri Venkateshwara College of Engineering, Bengaluru, March 2021

== Skills

#strong[NLP & AI Research:] Natural Language Processing, Transformers, Abstractive Summarization, RAG, LangChain, Edge AI, Crisis Informatics

#strong[Programming:] Python, C, Mojo, Perl, Julia, Rust, Lua

#strong[Data Science:] KNIME, Power BI, Tableau, BigQuery, Databricks, Pandas, Scikit-learn

#strong[Databases:] Oracle, MySQL, MongoDB, PostgreSQL

#strong[DevOps & Systems:] RedHat Linux, Ubuntu, Shell Scripting, Git, GitHub, Docker, Vim

== Languages

#strong[English:] Professional proficiency

#strong[Hindi & Marathi:] Native speaker
