// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-research",
          title: "research",
          description: "Research interests and contributions.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/research/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of your cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "Edit the `_data/repositories.yml` and change the `github_users` and `github_repos` lists to include your own GitHub profile and repositories.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-teaching",
          title: "teaching",
          description: "Courses taught across institutions including ICEM Pune, PCU Pune, SCOET Amravati, DRGITR Amravati, and Great Learning.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "Curriculum Vitae — Tushar Ravindra Mahore",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-certificates",
          title: "certificates",
          description: "Professional certifications, specializations, and training programs completed.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/certificates/";
          },
        },{id: "dropdown-bookshelf",
              title: "bookshelf",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/books/";
              },
            },{id: "dropdown-blog",
              title: "blog",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/blog/";
              },
            },{id: "post-building-a-rag-pipeline-from-scratch-with-python",
        
          title: "Building a RAG Pipeline from Scratch with Python",
        
        description: "A hands-on walkthrough of building a Retrieval-Augmented Generation system — from document indexing to grounded LLM responses — using FAISS, sentence-transformers, and a local model.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/rag-pipeline-from-scratch/";
          
        },
      },{id: "post-backpropagation-from-scratch-with-numpy",
        
          title: "Backpropagation from Scratch with NumPy",
        
        description: "Implement a two-layer neural network with forward pass, loss computation, and backpropagation using only NumPy. No frameworks — every gradient derived and coded explicitly.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/backpropagation-from-scratch-numpy/";
          
        },
      },{id: "post-text-classification-for-cyberbullying-detection-a-practical-nlp-walkthrough",
        
          title: "Text Classification for Cyberbullying Detection: A Practical NLP Walkthrough",
        
        description: "Build a cyberbullying detection classifier from raw text to evaluation — preprocessing, TF-IDF features, SVM and Random Forest, cross-validation, and a full sklearn pipeline. Based on real research published at IEEE ACROSET 2024.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/cyberbullying-detection-nlp-walkthrough/";
          
        },
      },{id: "post-explainable-ai-in-healthcare-tabtransformer-shap-for-iomt-security",
        
          title: "Explainable AI in Healthcare: TabTransformer + SHAP for IoMT Security",
        
        description: "Apply a TabTransformer — a Transformer architecture for tabular data — to IoMT intrusion detection, then explain its predictions using SHAP. A deep dive into XAI for high-stakes healthcare systems, based on our Academic Press 2026 book chapter.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/explainable-ai-healthcare-tabtransformer-shap/";
          
        },
      },{id: "post-google-gemini-updates-flash-1-5-gemma-2-and-project-astra",
        
          title: 'Google Gemini updates: Flash 1.5, Gemma 2 and Project Astra <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "We’re sharing updates across our Gemini family of models and a glimpse of Project Astra, our vision for the future of AI assistants.",
        section: "Posts",
        handler: () => {
          
            window.open("https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/", "_blank");
          
        },
      },{id: "post-displaying-external-posts-on-your-al-folio-blog",
        
          title: 'Displaying External Posts on Your al-folio Blog <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.open("https://medium.com/@al-folio/displaying-external-posts-on-your-al-folio-blog-b60a1d241a0a?source=rss-17feae71c3c4------2", "_blank");
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-enrolled-as-phd-student-at-symbiosis-institute-of-technology-sit-pune-researching-edge-deployable-nlp-pipelines-for-real-time-disaster-situation-summarization",
          title: 'Enrolled as PhD student at Symbiosis Institute of Technology (SIT), Pune, researching edge-deployable...',
          description: "",
          section: "News",},{id: "news-paper-presented-at-ieee-acroset-2024-cyberbullying-classification-using-nlp-and-machine-learning-techniques-acropolis-institute-of-technology-amp-amp-research-indore",
          title: 'Paper presented at IEEE ACROSET 2024 — Cyberbullying Classification Using NLP and Machine...',
          description: "",
          section: "News",},{id: "news-journal-paper-published-detection-of-multi-class-skin-cancer-using-sgd-augmentation-model-and-activation-mapping-journal-of-innovative-image-processing-vol-7-4-2025",
          title: 'Journal paper published — Detection of Multi-class Skin Cancer using SGD Augmentation Model...',
          description: "",
          section: "News",},{id: "news-book-chapter-published-an-explainable-hybrid-tabtransformer-random-forest-model-for-biometric-security-in-iomt-healthcare-systems-academic-press-2026",
          title: 'Book chapter published — An Explainable Hybrid TabTransformer–Random Forest Model for Biometric Security...',
          description: "",
          section: "News",},{id: "news-joined-indira-college-of-engineering-and-management-icem-pune-as-assistant-professor-in-the-department-of-ai-amp-amp-data-science",
          title: 'Joined Indira College of Engineering and Management (ICEM), Pune as Assistant Professor in...',
          description: "",
          section: "News",},{id: "projects-edge-nlp-pipeline-for-disaster-summarization",
          title: 'Edge NLP Pipeline for Disaster Summarization',
          description: "PhD research — offline, edge-deployable NLP system for real-time situation summarization during disasters using social media data.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-cyberbullying-detection-using-nlp-amp-ml",
          title: 'Cyberbullying Detection using NLP &amp;amp; ML',
          description: "Automated classification of cyberbullying content on social media using Natural Language Processing and Machine Learning. Published at IEEE ACROSET 2024.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-explainable-tabtransformer-rf-model-for-iomt-security",
          title: 'Explainable TabTransformer–RF Model for IoMT Security',
          description: "Hybrid explainable AI model combining TabTransformer and Random Forest for biometric security in IoMT healthcare systems. Published in Academic Press, 2026.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-multi-class-skin-cancer-detection-using-deep-learning",
          title: 'Multi-class Skin Cancer Detection using Deep Learning',
          description: "Deep learning system for multi-class skin cancer detection using SGD augmentation and activation mapping. Published in Journal of Innovative Image Processing, 2025.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-swarm-robotics-for-crisis-management",
          title: 'Swarm Robotics for Crisis Management',
          description: "Coordinated multi-robot response strategies for crisis scenarios using swarm intelligence and AI. Published in Auerbach Publications, 2025.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-ai-based-covid-19-omicron-early-detection-system",
          title: 'AI-Based Covid-19 / Omicron Early Detection System',
          description: "AI system for early symptom-based detection of Covid-19 and Omicron variants. Filed as Indian Patent, June 2022.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-media-analysis-system-for-positive-thought-broadcasting",
          title: 'Media Analysis System for Positive Thought Broadcasting',
          description: "AI-driven system to identify and surface positive, constructive media content during the Covid-19 pandemic. Registered Copyright, March 2021.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-credit-card-fraud-detection-using-ml-amp-deep-learning",
          title: 'Credit Card Fraud Detection using ML &amp;amp; Deep Learning',
          description: "Comparative study of ML and deep learning approaches for credit card fraud detection on imbalanced datasets. Published at Springer ICICC 2022 and AIP 2023.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-student-attendance-monitoring-via-facial-recognition",
          title: 'Student Attendance Monitoring via Facial Recognition',
          description: "Automated contactless student attendance system using deep learning-based facial recognition. Published at Springer ICICC 2022.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{id: "teachings-advanced-databases",
          title: 'Advanced Databases',
          description: "Advanced topics in database systems — distributed databases, query optimisation, spatial databases, data warehousing, and NewSQL systems.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/advanced-databases/";
            },},{id: "teachings-big-data-analytics",
          title: 'Big Data Analytics',
          description: "Processing and analysing large-scale datasets using Hadoop, Spark, and cloud platforms. Topics include MapReduce, distributed storage, streaming analytics, and visualisation at scale.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/big-data-analytics/";
            },},{id: "teachings-computer-programming-c-language",
          title: 'Computer Programming (C Language)',
          description: "Foundational programming course using C — covering structured programming, arrays, pointers, functions, structures, and file handling.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/computer-programming-c/";
            },},{id: "teachings-data-mining",
          title: 'Data Mining',
          description: "Techniques for discovering patterns, associations, and knowledge from large datasets. Covers classification, clustering, association rule mining, and anomaly detection.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-mining/";
            },},{id: "teachings-data-science-amp-statistics",
          title: 'Data Science &amp;amp; Statistics',
          description: "Covers the full data science workflow — statistical foundations, data wrangling, exploratory analysis, machine learning, and visualization using Python and R. Taught at PCU and Great Learning (2023–2025).",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals/";
            },},{id: "teachings-database-management-systems",
          title: 'Database Management Systems',
          description: "Relational database design, SQL, transaction management, normalisation, and an introduction to NoSQL systems.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/database-management-systems/";
            },},{id: "teachings-design-and-analysis-of-algorithms",
          title: 'Design and Analysis of Algorithms',
          description: "Algorithm design paradigms, complexity analysis, sorting and searching, graph algorithms, dynamic programming, greedy methods, and NP-completeness.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/design-analysis-algorithms/";
            },},{id: "teachings-devops-practical",
          title: 'DevOps (Practical)',
          description: "Hands-on DevOps practices — version control with Git, CI/CD pipelines, containerisation with Docker, and Linux system administration.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/devops/";
            },},{id: "teachings-discrete-structure-and-graph-theory",
          title: 'Discrete Structure and Graph Theory',
          description: "Mathematical foundations for computer science — logic, set theory, relations, functions, combinatorics, and graph theory.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/discrete-structures/";
            },},{id: "teachings-artificial-neural-networks-317531",
          title: 'Artificial Neural Networks (317531)',
          description: "T.E. AI &amp; Data Science, SPPU 2019 pattern. Covers biological foundations through modern deep architectures — perceptrons, backpropagation, CNNs, RNNs, and transfer learning.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning/";
            },},{id: "teachings-java-programming",
          title: 'Java Programming',
          description: "Core Java programming including OOP principles, collections, exception handling, multithreading, and an introduction to Java EE concepts.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/java-programming/";
            },},{id: "teachings-network-security",
          title: 'Network Security',
          description: "Principles and practices of securing computer networks — cryptography, authentication, firewalls, intrusion detection, and common attack vectors.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/network-security/";
            },},{id: "teachings-operations-research-and-management",
          title: 'Operations Research and Management',
          description: "Mathematical optimisation techniques for decision-making — linear programming, transportation and assignment problems, queuing theory, and simulation.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/operations-research/";
            },},{id: "teachings-python-programming",
          title: 'Python Programming',
          description: "Comprehensive Python programming — from core syntax and data structures to OOP, file handling, and application development. Taught across multiple institutions and online platforms.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/python-programming/";
            },},{id: "teachings-theory-of-computation",
          title: 'Theory of Computation',
          description: "Formal languages, automata theory, regular expressions, context-free grammars, Turing machines, decidability, and computational complexity.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/theory-of-computation/";
            },},{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/assets/pdf/example_pdf.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6D%61%68%6F%72%65%74%75%73%68%61%72@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/mahoretushar", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=nETPgMwAAAAJ", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0009-0000-9406-1178", "_blank");
        },
      },{
        id: 'social-scopus',
        title: 'Scopus',
        section: 'Socials',
        handler: () => {
          window.open("https://www.scopus.com/authid/detail.uri?authorId=57215925954", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
