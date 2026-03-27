import { GitHubIcon, LinkedInIcon, XIcon, HuggingFaceIcon } from "@/components/icons";

export const RESUME_DATA = {
  name: "Divi Eswar Chowdary",
  initials: "ED",
  location: "Andhra Pradesh, India",
  locationLink: "https://www.google.com/maps/place/Andhra+Pradesh/",
  about:
    "AI/ML Engineer · Computer Vision · LLMs · Agentic Systems",
  personalWebsiteUrl: "https://eswardivi.github.io/",
  summary:
    "AI/ML Engineer with expertise in computer vision, LLM-powered agentic systems, and production-grade deployment. Published researcher with hands-on experience building and scaling AI solutions across safety monitoring, enterprise intelligence, and NLP — from prototype to production.",
  avatarUrl: "https://avatars.githubusercontent.com/u/76403422?v=4",
  contact: {
    email: "eswar.divi.902@gmail.com",
    tel: null,
    social: [

      {
        name: "GitHub",
        url: "https://github.com/eswardivi",
        icon: GitHubIcon,
      },
      {
        name: "LinkedIn",
        url: "https://www.linkedin.com/in/eswardivi/",
        icon: LinkedInIcon,
      },
      {
        name: "X",
        url: "https://twitter.com/eswar_divi",
        icon: XIcon,
      },
      {
        name: "Hugging Face",
        url: "https://huggingface.co/eswardivi",
        icon: HuggingFaceIcon,
      }
    ],
  },
  education: [
    {
      school: "Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham",
      degree: "B.Tech in Computer Science and Engineering (Artificial Intelligence) · CGPA: 8.77/10",
      start: "2020",
      end: "2024",
    },
  ],
  work: [
    {
      company: "Schlumberger",
      link: "https://www.slb.com/",
      badges: ["Onsite"],
      title: "Data Scientist",
      start: "July 2024",
      end: "Present",
      description: [
        "Digital Factory – AI-Powered Safety & Operations: Engineered a computer vision platform monitoring compliance across 30+ facilities and 800+ cameras; fine-tuned YOLO models for PPE detection and forklift proximity alerts, deployed via TorchServe with torchao quantization for production inference.",
        "Resolved false positives in lifting compliance by integrating monocular depth estimation to correct perspective distortions; replaced frame-level classification with VideoMAE video classification, improving alert accuracy from 70% to 90%.",
        "People Finder – Enterprise Expert & Org Intelligence Agent: Co-designed an LLM-powered agentic system serving 1,000+ employees, enabling natural-language queries over reporting structures, skill directories, and org data; architected multi-step reasoning using LangChain and LangGraph with hybrid SQL + vector retrieval."
      ],
    },
    {
      company: "Schlumberger",
      link: "https://www.slb.com/",
      badges: ["Onsite"],
      title: "Data Scientist Intern",
      start: "June 2023",
      end: "August 2023",
      description: [
        "Supply Chain Intelligence: Built a multi-task deep learning model generating unified product embeddings from RFM attributes and metadata, enabling alternative product recommendations for supply chain decision-making.",
        "Designed a hybrid MLP-Transformer backbone applying self-attention over categorical and continuous RFM features; shared representations jointly optimized across multiple downstream tasks for robust per-product embeddings."
      ],
    },
  ],
  skills: [
    {
      category: "Languages",
      items: ["Python", "SQL"],
    },
    {
      category: "ML/DL Frameworks",
      items: ["PyTorch", "TensorFlow", "Scikit-learn", "HuggingFace Transformers", "timm", "YOLO", "VideoMAE", "TorchServe", "torchao"],
    },
    {
      category: "LLM & Agents",
      items: ["LLM Fine-tuning", "Model Merging", "LangChain", "LangGraph", "RAG", "Prompt Engineering"],
    },
    {
      category: "Tools & Deployment",
      items: ["Docker", "FastAPI", "Streamlit", "Gradio", "HuggingFace Spaces", "AWS SageMaker", "GCP", "Supabase", "pgvector"],
    },
  ],
  projects: [
    {
      title: "LumaBot: AI Customer Support SaaS",
      techStack: [
        "FastAPI",
        "PostgreSQL",
        "Qdrant",
        "LlamaIndex",
        "OpenAI",
        "Docker",
        "RAG",
      ],
      description:
        "Built a multitenant SaaS platform for deploying AI chatbots; architected a RAG pipeline with Qdrant, LlamaIndex, and OpenAI embeddings over a FastAPI + PostgreSQL backend with real-time streaming, async ingestion, and Docker deployment.",
      link: {
        label: "Live Demo",
        href: "https://lumabot.app/",
      },
    },
    {
      title: "MedGPT: Medical Diagnostic Chatbot",
      techStack: [
        "Python",
        "LLaMA-2",
        "QLoRA",
        "RAG",
        "Fine-tuning",
      ],
      description:
        "Fine-tuned LLaMA-2 (7B) using QLoRA on a custom ~10K medical Q&A dataset built by scraping and translating Korean medical sources; augmented with a RAG pipeline for grounded, citation-backed responses.",
    },
    {
      title: "ModelHub",
      techStack: [
        "Full Stack Developer",
        "JavaScript",
        "React",
        "Node.js",
        "Python",
      ],
      description:
        "ModelHub is a platform for sharing, discovering, and running machine learning models.",
      link: {
        label: "modelhub.vercel.app",
        href: "https://modelhub.vercel.app/",
      },
    },
    {
      title: "Podcastify",
      techStack: [
        "Python",
        "LLMs",
        "Streamlit"
      ],
      description:
        "Convert articles from URLs into listenable audio podcasts.",
      link: {
        label: "narrateit.streamlit.app",
        href: "https://narrateit.streamlit.app/",
      },
    },
    {
      title: "RosBot: Self-Driven Robot with Obstacle Avoidance",
      techStack: [
        "ROS",
        "TensorRT",
        "Resnet18",
        "OpenCV",
      ],
      description:
        "Designed and implemented RosBot, a self-driving robot with obstacle avoidance, using ROS, TensorRT, Resnet18, and OpenCV. Optimized Resnet18 model with TensorRT for accurate obstacle detection and utilized OpenCV for efficient image processing.",
      link: {
        label: "github.com",
        href: "https://github.com/EswarDivi/rosbot",
      },
    },
  ],
  researchPapers: [
    {
      title: "An analysis of data leakage and generalizability in MRI based classification of Parkinson's Disease using explainable 2D Convolutional Neural Networks",
      date: "2024",
      doilink: "https://doi.org/10.1016/j.dsp.2024.104407",
      description:
        "This paper presents a study on the detection of Parkinson's Disease (PD) from T1-weighted MRI scans using Convolutional Neural Networks (CNNs). The study investigates the potential for bias propagation in CNN models due to data leakage and evaluates the generalizability of the models to external datasets.",
    },
    {
      title: "Transformer-Based Multilingual Automatic Speech Recognition (ASR) Model for Dravidian Languages",
      date: "2024",
      doilink: "https://doi.org/10.1002/9781394214624.ch13",
      description:
        "This paper presents a approach to ASR for Dravidian languages like Tamil and Telugu. It builds upon the strengths of the powerful Whisper model, known for its multilingual capabilities, and fine-tunes it specifically for these under-resourced languages. This approach achieves significant improvements in WER compared to existing models.",
    },
    {
      title: "Going Beyond Traditional Methods: Using LSTM Networks to Predict Rainfall in Kerala.",
      date: "2024",
      doilink: "https://doi.org/10.1007/978-3-031-53717-2_11",
      description:
        "This paper presents an approach to forecast rainfall in Kerala using LSTM models. The study compares the performance of LSTM models with traditional time series forecasting methods and demonstrates the superior accuracy of LSTM models in predicting rainfall in the region.",
    }
  ],

} as const;
