# 🤖 HuggingFace AI Pipeline Demo

A demonstration of advanced AI text processing using HuggingFace Transformers and LangChain, optimized for Apple Silicon (M1/M2/M3/M4) MacBooks.

## 🚀 What This Project Does

This project creates an intelligent text processing pipeline that:
1. **Summarizes** long text using state-of-the-art models
2. **Refines** summaries for better quality
3. **Answers questions** about the content
4. **Chains models together** using LangChain for complex workflows

## 🧠 Key Concepts & Technologies

### 🤗 HuggingFace Transformers
**What it is:** The leading library for pre-trained transformer models
**Why we use it:** 
- Access to thousands of pre-trained models
- Easy-to-use pipeline interface
- Excellent performance on various NLP tasks
- Active community and regular updates

### 🦜 LangChain
**What it is:** Framework for developing applications with large language models
**Why we use it:**
- **Chaining**: Connect multiple AI models in sequence
- **Templates**: Create reusable prompt templates
- **Abstraction**: Simplifies complex AI workflows
- **Flexibility**: Easy to modify and extend pipelines

### 🧠 Transformer Models Used

#### BART (Bidirectional and Auto-Regressive Transformers)
- **Model**: `facebook/bart-large-cnn`
- **Purpose**: Text summarization
- **Why BART**: Specifically fine-tuned for summarization tasks, excellent at preserving key information

#### RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Model**: `deepset/roberta-base-squad2`
- **Purpose**: Question answering
- **Why RoBERTa**: Superior performance on reading comprehension tasks, trained on SQuAD dataset

### ⚡ PyTorch & Apple Silicon Optimization
**What it is:** Deep learning framework with Metal Performance Shaders (MPS) support
**Why we use it:**
- **MPS Device**: Leverages Apple Silicon GPU acceleration
- **Memory Efficiency**: Optimized for M-series chip architecture
- **Performance**: Significant speedup compared to CPU-only processing

## 🏗️ Architecture

```
User Input Text
       ↓
[Summary Template] ← User specifies length (short/medium/long)
       ↓
[BART Summarizer] ← facebook/bart-large-cnn
       ↓
[BART Refiner] ← facebook/bart-large
       ↓
Final Summary
```

## 📦 Dependencies Explained

### Core AI Libraries
- **transformers**: HuggingFace's transformer models library
- **torch**: PyTorch deep learning framework
- **accelerate**: Hardware acceleration for large models

### LangChain Ecosystem
- **langchain**: Core LangChain framework
- **langchain-huggingface**: HuggingFace integration for LangChain
- **langchain-core**: Core LangChain components and abstractions

### Model-Specific Requirements
- **sentencepiece**: Tokenization library (required for some models)
- **protobuf**: Protocol buffers (required for model serialization)
- **safetensors**: Safe tensor serialization format

### Performance & Utility
- **numpy**: Numerical computing
- **tqdm**: Progress bars for model downloads
- **requests**: HTTP library for model downloads

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- macOS with Apple Silicon (M1/M2/M3/M4) recommended
- 8GB+ RAM (16GB+ recommended for larger models)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/shree-bd/huggingface-demo.git
cd huggingface-demo
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Pipeline
```bash
# Activate virtual environment
source venv/bin/activate

# Run the main script
python main.py
```

### Interactive Usage
1. **Enter text to summarize** - Paste any long article or text
2. **Choose summary length** - Type "short", "medium", or "long"
3. **Get AI-generated summary** - The pipeline will process and return a refined summary

### Example Input
```
The concept of artificial intelligence has evolved dramatically since its inception in the 1950s, transforming from theoretical discussions among computer scientists to practical applications that now permeate virtually every aspect of modern life...
```

### Example Output
```
🤖 **Generated Summary:**
Artificial intelligence has evolved from 1950s theoretical concepts to practical applications across industries. Key developments include expert systems in the 1970s-80s, machine learning resurgence in the 1990s-2000s, and the deep learning revolution of the 2010s. Modern AI uses transformer architectures for language processing and is deployed in healthcare, finance, transportation, and entertainment, while raising important ethical considerations for future development.
```

## 🎯 Key Features

- **✅ Apple Silicon Optimized**: Uses MPS device for M-series chip acceleration
- **✅ Multiple Model Chain**: Combines summarization and refinement models
- **✅ Interactive Interface**: User-friendly command-line interaction
- **✅ Flexible Length Control**: Adjustable summary lengths
- **✅ Production Ready**: Proper error handling and optimization

## 🔧 Customization

### Using Different Models
Replace model names in `main.py`:
```python
# For different summarization models
summarization_pipeline = pipeline(task="summarization", model="t5-base", device="mps")

# For different QA models
qa_pipeline = pipeline(task="question-answering", model="bert-base-uncased", device="mps")
```

### Modifying Templates
Update the prompt template:
```python
summary_template = PromptTemplate.from_template(
    "Create a {length} summary focusing on {aspect}:\n\n{text}"
)
```

## 🧪 Testing

Test with the provided sample text or use your own content:
- News articles
- Research papers
- Blog posts
- Documentation

## 📊 Performance Notes

- **First run**: Models will download (~2-3GB total)
- **Subsequent runs**: Much faster as models are cached
- **Apple Silicon**: Significant performance boost with MPS
- **Memory usage**: ~4-8GB RAM depending on text length

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **HuggingFace** for providing excellent pre-trained models
- **LangChain** for the powerful chaining framework
- **PyTorch** for Apple Silicon optimization
- **Meta AI** for BART models
- **deepset** for RoBERTa QA models

## 📚 Further Reading

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Apple Silicon ML Optimization](https://developer.apple.com/machine-learning/)
- [BART Paper](https://arxiv.org/abs/1910.13461)
- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)

---

*Built with ❤️ for the AI community*
