"""Deterministic local role-profile knowledge base.

Each profile defines structured expectations for a supported role
using ``JobDescription``-compatible concepts: required skills, preferred
skills, experience expectations, education/qualifications, and
responsibilities. Profiles are local data only — no external calls,
no network dependency, no paid services, and crucially NO Ollama/LLM is
used to invent requirements.

Role aliases are normalised the same way as skill aliases (case,
whitespace, unicode) so that lookups are forgiving. Aliases resolve to
a canonical profile — there are never two profiles for the same role.

Categories covered:
  AI / ML, Software Development, Cloud / DevOps, Cybersecurity,
  Data / Database, QA / Testing, Systems / Infrastructure, plus the
  existing Product/UX profiles.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class RoleProfile:
    title: str
    aliases: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    preferred_skills: tuple[str, ...] = ()
    experience_requirements: tuple[str, ...] = ()
    education_requirements: tuple[str, ...] = ()
    qualifications: tuple[str, ...] = ()
    responsibilities: tuple[str, ...] = ()


SUPPORTED_ROLES: tuple[RoleProfile, ...] = (
    # --- AI / ML ---------------------------------------------------------
    RoleProfile(
        title="AI Engineer",
        aliases=(
            "artificial intelligence engineer",
            "ai/ml engineer",
            "ai ml engineer",
            "ai engineer",
        ),
        required_skills=(
            "Python",
            "Machine Learning",
            "Statistics",
            "SQL",
        ),
        preferred_skills=(
            "TensorFlow",
            "PyTorch",
            "Docker",
            "AWS",
            "CI/CD",
            "Git",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and deploy machine learning-powered applications",
            "Build and productionize ML pipelines",
            "Collaborate with data scientists and ML engineers",
        ),
    ),
    RoleProfile(
        title="Machine Learning Engineer",
        aliases=("ml engineer",),
        required_skills=(
            "Python",
            "Machine Learning",
            "Statistics",
            "SQL",
        ),
        preferred_skills=(
            "TensorFlow",
            "PyTorch",
            "Scikit-learn",
            "Pandas",
            "Docker",
            "AWS",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and deploy machine learning models",
            "Create robust ML training and serving pipelines",
            "Monitor model performance in production",
        ),
    ),
    RoleProfile(
        title="Deep Learning Engineer",
        aliases=("deep learning engineer", "dl engineer"),
        required_skills=(
            "Python",
            "Deep Learning",
            "Mathematics",
            "Neural Networks",
            "SQL",
        ),
        preferred_skills=(
            "TensorFlow",
            "PyTorch",
            "CUDA",
            "GPU programming",
            "Docker",
            "AWS",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and implement deep neural networks",
            "Optimize models for inference performance",
            "Leverage GPU acceleration for training and inference",
        ),
    ),
    RoleProfile(
        title="NLP Engineer",
        aliases=("natural language processing engineer",),
        required_skills=(
            "Python",
            "Natural Language Processing",
            "Machine Learning",
            "Text Processing",
        ),
        preferred_skills=(
            "Transformers",
            "spaCy",
            "NLTK",
            "Hugging Face",
            "PyTorch",
            "TensorFlow",
            "Docker",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build NLP pipelines and language models",
            "Preprocess and tokenize text data",
            "Fine-tune transformer models for domain-specific tasks",
        ),
    ),
    RoleProfile(
        title="Computer Vision Engineer",
        aliases=("computer vision engineer", "cv engineer"),
        required_skills=(
            "Python",
            "Computer Vision",
            "Machine Learning",
            "Image Processing",
        ),
        preferred_skills=(
            "OpenCV",
            "TensorFlow",
            "PyTorch",
            "Keras",
            "CUDA",
            "Docker",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Develop computer vision models and pipelines",
            "Process and annotate image/video data",
            "Deploy vision models to edge and cloud",
        ),
    ),
    RoleProfile(
        title="Generative AI Engineer",
        aliases=("genai engineer",),
        required_skills=(
            "Python",
            "Generative AI",
            "Machine Learning",
            "Deep Learning",
        ),
        preferred_skills=(
            "LLMs",
            "Transformers",
            "Prompt Engineering",
            "LangChain",
            "PyTorch",
            "TensorFlow",
            "Docker",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and fine-tune generative AI applications",
            "Engineer prompts and LLM pipelines",
            "Integrate generative models into products",
        ),
    ),
    RoleProfile(
        title="AI Research Engineer",
        aliases=("ai researcher",),
        required_skills=(
            "Python",
            "Machine Learning",
            "Mathematics",
            "Statistics",
        ),
        preferred_skills=(
            "PyTorch",
            "TensorFlow",
            "Research",
            "Paper Reading",
            "LaTeX",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Master's degree",),
        qualifications=(),
        responsibilities=(
            "Conduct applied research on ML/AI problems",
            "Read and reproduce academic papers",
            "Publish findings and prototype novel methods",
        ),
    ),
    RoleProfile(
        title="MLOps Engineer",
        aliases=("mlops",),
        required_skills=(
            "Python",
            "Machine Learning",
            "Docker",
            "CI/CD",
            "Kubernetes",
        ),
        preferred_skills=(
            "Terraform",
            "MLflow",
            "Airflow",
            "AWS",
            "GCP",
            "Prometheus",
            "Grafana",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain ML model deployment pipelines",
            "Ensure reproducibility and versioning of ML artifacts",
            "Monitor and alert on model and data drift",
        ),
    ),
    RoleProfile(
        title="Data Scientist",
        aliases=("data scientist", "data science"),
        required_skills=(
            "Python",
            "Statistics",
            "Machine Learning",
            "SQL",
            "Pandas",
        ),
        preferred_skills=(
            "TensorFlow",
            "PyTorch",
            "Spark",
            "Docker",
            "AWS",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Master's degree",),
        qualifications=(),
        responsibilities=(
            "Build predictive models from complex datasets",
            "Analyze business data and communicate findings",
            "Experiment with statistical methods and ML algorithms",
        ),
    ),
    RoleProfile(
        title="Data Engineer",
        aliases=("data engineer", "etl engineer"),
        required_skills=(
            "Python",
            "SQL",
            "ETL",
            "Data Pipelines",
        ),
        preferred_skills=(
            "Spark",
            "Kafka",
            "Airflow",
            "Hadoop",
            "Snowflake",
            "AWS",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain data ingestion and transformation pipelines",
            "Design and optimize data warehouses and lakes",
            "Ensure data reliability and quality",
        ),
    ),
    RoleProfile(
        title="Data Analyst",
        aliases=("data analyst",),
        required_skills=(
            "SQL",
            "Statistics",
            "Excel",
            "Data Visualization",
        ),
        preferred_skills=(
            "Python",
            "R",
            "Tableau",
            "Power BI",
            "Pandas",
        ),
        experience_requirements=("1+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Analyze datasets to extract actionable business insights",
            "Build dashboards and reports for stakeholders",
            "Clean and transform data for analysis",
        ),
    ),

    # --- Software Development -------------------------------------------
    RoleProfile(
        title="Software Engineer",
        aliases=(
            "software developer",
            "developer",
            "swe",
        ),
        required_skills=(
            "Python",
            "JavaScript",
            "Git",
            "SQL",
        ),
        preferred_skills=(
            "Docker",
            "AWS",
            "React",
            "Node.js",
            "CI/CD",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and implement software solutions",
            "Write clean, maintainable code",
            "Collaborate with cross-functional teams",
        ),
    ),
    RoleProfile(
        title="Backend Developer",
        aliases=(
            "backend engineer",
            "server-side engineer",
            "api engineer",
        ),
        required_skills=(
            "Python",
            "SQL",
            "REST APIs",
            "Git",
        ),
        preferred_skills=(
            "Docker",
            "AWS",
            "PostgreSQL",
            "Node.js",
            "Redis",
            "CI/CD",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain server-side application logic",
            "Design and optimize relational databases",
            "Create and maintain RESTful APIs",
        ),
    ),
    RoleProfile(
        title="Frontend Developer",
        aliases=(
            "frontend engineer",
            "front-end developer",
            "ui developer",
        ),
        required_skills=(
            "HTML",
            "CSS",
            "JavaScript",
            "React",
        ),
        preferred_skills=(
            "TypeScript",
            "Vue",
            "Angular",
            "Jest",
            "Git",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build responsive user interfaces",
            "Translate designs into clean code",
            "Write unit and integration tests",
        ),
    ),
    RoleProfile(
        title="Full Stack Developer",
        aliases=("full stack developer", "fullstack developer", "full stack engineer"),
        required_skills=(
            "HTML",
            "CSS",
            "JavaScript",
            "SQL",
            "Git",
        ),
        preferred_skills=(
            "React",
            "Node.js",
            "Python",
            "Docker",
            "AWS",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build end-to-end web applications",
            "Work across frontend and backend codebases",
            "Design and maintain databases",
        ),
    ),
    RoleProfile(
        title="Python Developer",
        aliases=("python engineer",),
        required_skills=(
            "Python",
            "SQL",
            "Git",
        ),
        preferred_skills=(
            "Django",
            "Flask",
            "FastAPI",
            "Pandas",
            "Docker",
            "AWS",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain Python applications and services",
            "Write reusable, testable, and efficient code",
            "Integrate Python with databases and APIs",
        ),
    ),
    RoleProfile(
        title="Java Developer",
        aliases=("java engineer",),
        required_skills=(
            "Java",
            "SQL",
            "Git",
        ),
        preferred_skills=(
            "Spring",
            "Maven",
            "JUnit",
            "Docker",
            "AWS",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain Java applications",
            "Design object-oriented systems and services",
            "Write and maintain unit tests",
        ),
    ),
    RoleProfile(
        title="C++ Developer",
        aliases=("cpp developer",),
        required_skills=(
            "C++",
            "Git",
            "SQL",
        ),
        preferred_skills=(
            "STL",
            "Boost",
            "CMake",
            "Docker",
            "Linux",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain high-performance C++ applications",
            "Work with systems-level and performance-critical code",
            "Write efficient, testable C++ code",
        ),
    ),
    RoleProfile(
        title="JavaScript Developer",
        aliases=("js developer",),
        required_skills=(
            "JavaScript",
            "HTML",
            "CSS",
            "Git",
        ),
        preferred_skills=(
            "React",
            "Vue",
            "Node.js",
            "TypeScript",
            "Webpack",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build interactive frontend applications",
            "Write maintainable JavaScript code",
            "Integrate with REST APIs",
        ),
    ),
    RoleProfile(
        title="TypeScript Developer",
        aliases=("ts developer",),
        required_skills=(
            "TypeScript",
            "JavaScript",
            "Git",
        ),
        preferred_skills=(
            "React",
            "Node.js",
            "Angular",
            "NestJS",
            "Docker",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build typed frontend and backend applications",
            "Write robust, type-safe code",
            "Refactor and maintain large codebases",
        ),
    ),
    RoleProfile(
        title="Mobile Developer",
        aliases=("mobile app developer",),
        required_skills=(
            "Mobile Development",
            "Git",
        ),
        preferred_skills=(
            "React Native",
            "Flutter",
            "Swift",
            "Kotlin",
            "Dart",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain mobile applications",
            "Collaborate with designers and backend teams",
            "Optimize performance across platforms",
        ),
    ),
    RoleProfile(
        title="Android Developer",
        aliases=("android engineer",),
        required_skills=(
            "Kotlin",
            "Java",
            "Android",
            "Git",
        ),
        preferred_skills=(
            "Jetpack Compose",
            "Dagger",
            "Retrofit",
            "Firebase",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain Android applications",
            "Write efficient, well-tested Kotlin code",
            "Integrate with backend APIs",
        ),
    ),
    RoleProfile(
        title="iOS Developer",
        aliases=("iphone developer", "swift developer"),
        required_skills=(
            "Swift",
            "iOS",
            "Git",
        ),
        preferred_skills=(
            "SwiftUI",
            "Objective-C",
            "Xcode",
            "Combine",
            "Firebase",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain iOS applications",
            "Write efficient, well-tested Swift code",
            "Integrate with backend APIs",
        ),
    ),

    # --- Cloud / DevOps -----------------------------------------------
    RoleProfile(
        title="DevOps Engineer",
        aliases=("devops",),
        required_skills=(
            "Linux",
            "Docker",
            "Kubernetes",
            "CI/CD",
            "Python",
            "Git",
        ),
        preferred_skills=(
            "AWS",
            "Terraform",
            "Ansible",
            "Prometheus",
            "Grafana",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Maintain CI/CD pipelines",
            "Ensure system reliability and scalability",
            "Automate infrastructure provisioning",
        ),
    ),
    RoleProfile(
        title="Cloud Engineer",
        aliases=("cloud infrastructure engineer",),
        required_skills=(
            "Cloud Computing",
            "AWS",
            "Terraform",
        ),
        preferred_skills=(
            "Docker",
            "Kubernetes",
            "GCP",
            "Azure",
            "CI/CD",
            "Python",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and manage cloud infrastructure",
            "Deploy and scale cloud-native applications",
            "Implement infrastructure as code",
        ),
    ),
    RoleProfile(
        title="AWS Engineer",
        aliases=("aws engineer", "amazon web services engineer"),
        required_skills=(
            "AWS",
            "Cloud Computing",
        ),
        preferred_skills=(
            "Terraform",
            "Docker",
            "Kubernetes",
            "CI/CD",
            "Python",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and manage AWS-based infrastructure and services",
            "Optimize costs and performance on AWS",
            "Automate AWS deployments",
        ),
    ),
    RoleProfile(
        title="Azure Engineer",
        aliases=("azure engineer", "microsoft azure engineer"),
        required_skills=(
            "Azure",
            "Cloud Computing",
        ),
        preferred_skills=(
            "Terraform",
            "Docker",
            "Kubernetes",
            "CI/CD",
            "Python",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and manage Azure-based infrastructure and services",
            "Optimize costs and performance on Azure",
            "Automate Azure deployments",
        ),
    ),
    RoleProfile(
        title="Google Cloud Engineer",
        aliases=("gcp engineer", "google cloud platform engineer"),
        required_skills=(
            "GCP",
            "Google Cloud",
            "Cloud Computing",
        ),
        preferred_skills=(
            "Terraform",
            "Docker",
            "Kubernetes",
            "CI/CD",
            "Python",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and manage GCP-based infrastructure and services",
            "Optimize costs and performance on GCP",
            "Automate GCP deployments",
        ),
    ),
    RoleProfile(
        title="Site Reliability Engineer",
        aliases=("sre",),
        required_skills=(
            "Linux",
            "Python",
            "Monitoring",
            "CI/CD",
        ),
        preferred_skills=(
            "Kubernetes",
            "Docker",
            "Prometheus",
            "Grafana",
            "SRE",
            "AWS",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Ensure reliability and scalability of production systems",
            "Write automation to replace manual toil",
            "Monitor and respond to system incidents",
        ),
    ),
    RoleProfile(
        title="Platform Engineer",
        aliases=("platform developer",),
        required_skills=(
            "Kubernetes",
            "Docker",
            "Infrastructure as Code",
            "CI/CD",
        ),
        preferred_skills=(
            "Terraform",
            "Helm",
            "Go",
            "Python",
            "AWS",
            "GCP",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain internal developer platforms",
            "Create reusable infrastructure and tooling",
            "Enable self-service for engineering teams",
        ),
    ),
    RoleProfile(
        title="Cloud Architect",
        aliases=("solution architect",),
        required_skills=(
            "Cloud Architecture",
            "AWS",
            "Azure",
        ),
        preferred_skills=(
            "GCP",
            "Docker",
            "Kubernetes",
            "Terraform",
            "Security",
            "Migration",
        ),
        experience_requirements=("5+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design multi-region cloud architectures",
            "Define cloud strategy and best practices",
            "Oversee cloud migrations and security",
        ),
    ),

    # --- Cybersecurity -----------------------------------------------
    RoleProfile(
        title="Cybersecurity Analyst",
        aliases=("cyber security analyst", "infosec analyst"),
        required_skills=(
            "Security",
            "Vulnerability Assessment",
            "SIEM",
        ),
        preferred_skills=(
            "Threat Analysis",
            "Incident Response",
            "Linux",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Monitor and analyze security alerts",
            "Conduct vulnerability assessments",
            "Respond to security incidents",
        ),
    ),
    RoleProfile(
        title="Cybersecurity Engineer",
        aliases=("cyber security engineer",),
        required_skills=(
            "Security Engineering",
            "Cryptography",
            "Identity Management",
        ),
        preferred_skills=(
            "SIEM",
            "Firewalls",
            "IAM",
            "Zero Trust",
            "Python",
            "Linux",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and implement security controls",
            "Build and maintain security tooling",
            "Conduct security reviews and audits",
        ),
    ),
    RoleProfile(
        title="Security Engineer",
        aliases=("information security engineer",),
        required_skills=(
            "Security",
            "Application Security",
            "Vulnerability Assessment",
        ),
        preferred_skills=(
            "OWASP",
            "Penetration Testing",
            "CI/CD Security",
            "SAST",
            "DAST",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain security infrastructure",
            "Conduct vulnerability assessments and penetration tests",
            "Implement security at the application layer",
        ),
    ),
    RoleProfile(
        title="SOC Analyst",
        aliases=("security operations center analyst",),
        required_skills=(
            "SIEM",
            "Security Monitoring",
            "Incident Response",
        ),
        preferred_skills=(
            "Splunk",
            "ELK",
            "Threat Intelligence",
            "Forensics",
        ),
        experience_requirements=("1+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Monitor security dashboards for alerts",
            "Investigate and triage security events",
            "Document and escalate incidents",
        ),
    ),
    RoleProfile(
        title="Penetration Tester",
        aliases=("pen tester", "ethical hacker", "penetration testing engineer"),
        required_skills=(
            "Penetration Testing",
            "Security Testing",
            "Kali Linux",
        ),
        preferred_skills=(
            "Metasploit",
            "Burp Suite",
            "OWASP",
            "Network Security",
            "Python",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Conduct authorized penetration tests",
            "Identify and document security vulnerabilities",
            "Write penetration testing reports",
        ),
    ),
    RoleProfile(
        title="Application Security Engineer",
        aliases=("appsec engineer",),
        required_skills=(
            "Application Security",
            "SAST",
            "DAST",
        ),
        preferred_skills=(
            "OWASP",
            "Penetration Testing",
            "CI/CD Security",
            "Threat Modeling",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Integrate security into the SDLC",
            "Review code and architecture for vulnerabilities",
            "Build security automation and tooling",
        ),
    ),
    RoleProfile(
        title="Cloud Security Engineer",
        aliases=(),
        required_skills=(
            "Cloud Security",
            "AWS Security",
            "Identity Management",
        ),
        preferred_skills=(
            "GCP Security",
            "Azure Security",
            "SIEM",
            "IAM",
            "Vulnerability Assessment",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Secure cloud infrastructure and workloads",
            "Implement cloud security best practices",
            "Conduct cloud security assessments",
        ),
    ),
    RoleProfile(
        title="Security Architect",
        aliases=(),
        required_skills=(
            "Security Architecture",
            "Risk Management",
            "Cryptography",
        ),
        preferred_skills=(
            "IAM",
            "Zero Trust",
            "Security Frameworks",
            "Compliance",
        ),
        experience_requirements=("5+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design enterprise-wide security architectures",
            "Define security standards and guidelines",
            "Advise on risk and compliance strategy",
        ),
    ),

    # --- Data / Database ----------------------------------------------
    RoleProfile(
        title="Database Administrator",
        aliases=("dba",),
        required_skills=(
            "SQL",
            "PostgreSQL",
            "MySQL",
            "Database Administration",
        ),
        preferred_skills=(
            "Oracle",
            "MongoDB",
            "Redis",
            "Backup and Recovery",
            "Performance Tuning",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Install, configure, and maintain database systems",
            "Monitor performance and troubleshoot issues",
            "Implement backup and disaster recovery",
        ),
    ),
    RoleProfile(
        title="Database Developer",
        aliases=(),
        required_skills=(
            "SQL",
            "PostgreSQL",
            "Stored Procedures",
        ),
        preferred_skills=(
            "Python",
            "ETL",
            "Database Design",
            "Performance Tuning",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and implement database schemas",
            "Write stored procedures and queries",
            "Optimize database performance",
        ),
    ),
    RoleProfile(
        title="BI Developer",
        aliases=("business intelligence developer",),
        required_skills=(
            "Business Intelligence",
            "SQL",
            "Data Warehousing",
        ),
        preferred_skills=(
            "Power BI",
            "Tableau",
            "SSRS",
            "ETL",
            "SSAS",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build BI dashboards and reports",
            "Design and maintain data warehouse models",
            "Transform raw data into business insights",
        ),
    ),
    RoleProfile(
        title="Business Intelligence Analyst",
        aliases=("bi analyst",),
        required_skills=(
            "Business Intelligence",
            "Data Analysis",
            "SQL",
        ),
        preferred_skills=(
            "Power BI",
            "Tableau",
            "Excel",
            "Dashboard Design",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Analyze business data and build reports",
            "Create dashboards for stakeholder consumption",
            "Identify business trends and insights",
        ),
    ),
    RoleProfile(
        title="Data Architect",
        aliases=(),
        required_skills=(
            "Data Architecture",
            "SQL",
            "Data Modeling",
        ),
        preferred_skills=(
            "Snowflake",
            "Redshift",
            "BigQuery",
            "ETL",
            "Data Governance",
        ),
        experience_requirements=("5+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design enterprise data architectures",
            "Define data models and governance standards",
            "Oversee data platform strategy",
        ),
    ),

    # --- QA / Testing -----------------------------------------------
    RoleProfile(
        title="QA Engineer",
        aliases=("qa", "quality assurance engineer"),
        required_skills=(
            "Testing",
            "QA",
            "Test Cases",
        ),
        preferred_skills=(
            "Selenium",
            "JUnit",
            "pytest",
            "CI/CD",
            "Automation",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and execute test plans",
            "Maintain test automation suites",
            "Report and track defects",
        ),
    ),
    RoleProfile(
        title="QA Analyst",
        aliases=("quality assurance analyst",),
        required_skills=(
            "Testing",
            "QA",
            "Test Cases",
        ),
        preferred_skills=(
            "Manual Testing",
            "Bug Reporting",
            "JIRA",
            "Test Plans",
        ),
        experience_requirements=("1+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Perform manual and automated testing",
            "Document test cases and scenarios",
            "Report defects with clear reproduction steps",
        ),
    ),
    RoleProfile(
        title="Software Test Engineer",
        aliases=("test engineer",),
        required_skills=(
            "Testing",
            "QA",
            "Test Automation",
        ),
        preferred_skills=(
            "Selenium",
            "JUnit",
            "pytest",
            "CI/CD",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and execute automated test suites",
            "Build test harnesses and frameworks",
            "Collaborate with developers on quality",
        ),
    ),
    RoleProfile(
        title="Automation Test Engineer",
        aliases=("automation engineer",),
        required_skills=(
            "Test Automation",
            "Selenium",
            "CI/CD",
        ),
        preferred_skills=(
            "pytest",
            "JUnit",
            "Jenkins",
            "API Testing",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain test automation frameworks",
            "Automate regression test suites",
            "Integrate tests into CI/CD pipelines",
        ),
    ),
    RoleProfile(
        title="SDET",
        aliases=("software development engineer in test",),
        required_skills=(
            "Test Automation",
            "SDET",
            "QA",
        ),
        preferred_skills=(
            "Selenium",
            "JUnit",
            "pytest",
            "CI/CD",
            "API Testing",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Write production-quality test code",
            "Build scalable test automation frameworks",
            "Advocate for quality across the team",
        ),
    ),

    # --- Systems / Infrastructure ------------------------------------
    RoleProfile(
        title="Systems Engineer",
        aliases=("systems engineering",),
        required_skills=(
            "Linux",
            "Networking",
            "System Design",
        ),
        preferred_skills=(
            "Docker",
            "Kubernetes",
            "Python",
            "Bash",
            "Ansible",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and maintain complex systems",
            "Troubleshoot system-level issues",
            "Coordinate with development and operations teams",
        ),
    ),
    RoleProfile(
        title="Systems Administrator",
        aliases=("sysadmin", "system administrator"),
        required_skills=(
            "Linux",
            "System Administration",
            "Bash",
        ),
        preferred_skills=(
            "Ansible",
            "Puppet",
            "Docker",
            "Monitoring",
            "Python",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Configure and maintain Linux servers",
            "Monitor system performance and uptime",
            "Implement security patches and updates",
        ),
    ),
    RoleProfile(
        title="Network Engineer",
        aliases=("networking engineer",),
        required_skills=(
            "Networking",
            "TCP/IP",
            "Firewalls",
        ),
        preferred_skills=(
            "Cisco",
            "Routing",
            "Switching",
            "Load Balancers",
            "VPN",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design and maintain network infrastructure",
            "Configure routers, switches, and firewalls",
            "Troubleshoot network connectivity and performance",
        ),
    ),
    RoleProfile(
        title="Network Administrator",
        aliases=("network admin",),
        required_skills=(
            "Networking",
            "TCP/IP",
        ),
        preferred_skills=(
            "Cisco",
            "DNS",
            "DHCP",
            "VPN",
            "Firewall Management",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Administer and maintain network systems",
            "Monitor network performance and uptime",
            "Manage IP addressing and DNS",
        ),
    ),
    RoleProfile(
        title="Infrastructure Engineer",
        aliases=("infra engineer",),
        required_skills=(
            "Linux",
            "Infrastructure as Code",
            "Networking",
        ),
        preferred_skills=(
            "Terraform",
            "Ansible",
            "Docker",
            "Kubernetes",
            "AWS",
        ),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Build and maintain infrastructure as code",
            "Automate provisioning and configuration",
            "Ensure infrastructure reliability and scalability",
        ),
    ),
    RoleProfile(
        title="Linux Administrator",
        aliases=("linux admin", "linux systems administrator"),
        required_skills=(
            "Linux",
            "Bash",
            "System Administration",
        ),
        preferred_skills=(
            "Ansible",
            "Docker",
            "Shell Scripting",
            "Monitoring",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Configure and maintain Linux servers",
            "Write shell scripts for automation",
            "Monitor system health and security",
        ),
    ),

    # --- Product / UX (existing profiles, retained) -------------------
    RoleProfile(
        title="Product Manager",
        aliases=("pm", "product owner", "product lead"),
        required_skills=(
            "Communication",
            "Agile",
            "Stakeholder Management",
        ),
        preferred_skills=("Analytics", "SQL", "Figma", "Jira"),
        experience_requirements=("3+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Define product vision and strategy",
            "Prioritize features and manage backlogs",
            "Collaborate with engineering and design teams",
        ),
    ),
    RoleProfile(
        title="UX Designer",
        aliases=(
            "ui designer",
            "user experience designer",
            "product designer",
            "designer",
        ),
        required_skills=("Figma", "User Research", "Wireframing", "Prototyping"),
        preferred_skills=(
            "HTML",
            "CSS",
            "JavaScript",
            "Accessibility",
            "Usability Testing",
        ),
        experience_requirements=("2+ years",),
        education_requirements=("Bachelor's degree",),
        qualifications=(),
        responsibilities=(
            "Design user interfaces and interactions",
            "Conduct user research and usability testing",
            "Create wireframes and prototypes",
        ),
    ),
)


_normaliser_cache: dict[str, str] = {}
_NORMALISE_RE = re.compile(r"\s+")


def _normalise(value: str) -> str:
    """Normalise a string for profile lookup (case-insensitive, trimmed)."""
    if value in _normaliser_cache:
        return _normaliser_cache[value]
    text = unicodedata.normalize("NFKC", value).strip()
    text = text.lower()
    text = _NORMALISE_RE.sub(" ", text)
    text = text.strip(" .,;:")
    _normaliser_cache[value] = text
    return text


def resolve_role_title(title: str) -> RoleProfile | None:
    """Resolve a job role title to a local profile, or None if unsupported.

    Matching is case-insensitive and supports aliases from each profile.
    """
    normalised = _normalise(title)
    if not normalised:
        return None
    for profile in SUPPORTED_ROLES:
        if _normalise(profile.title) == normalised:
            return profile
        for alias in profile.aliases:
            if _normalise(alias) == normalised:
                return profile
    return None


def get_supported_titles() -> list[str]:
    """Return the primary titles of all supported roles."""
    return [p.title for p in SUPPORTED_ROLES]


def search_roles(query: str) -> list[RoleProfile]:
    """Return supported profiles whose title or aliases contain the query.

    Matching is case-insensitive and normalised. The query may be a
    substring, so ``ml`` matches both *Machine Learning Engineer* and
    *MLOps Engineer*. An empty query returns nothing.
    """
    needle = _normalise(query)
    if not needle:
        return []
    results: list[RoleProfile] = []
    for profile in SUPPORTED_ROLES:
        haystack = _normalise(profile.title)
        if needle in haystack or needle in profile.title.lower():
            results.append(profile)
            continue
        for alias in profile.aliases:
            if needle in _normalise(alias) or needle in alias.lower():
                results.append(profile)
                break
    return results
