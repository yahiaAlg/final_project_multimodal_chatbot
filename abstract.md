# Abstract:

While Multi-modal Large Language Models (MM-LLMs) have made exciting progress recently, their ability to produce content in multiple modalities is limited, and they are primarily limited to input-side multi-modal understanding. Since humans perceive the world and interact with others through a variety of modalities, creating any-to-any MM-LLMs that can receive and deliver content in any modality is crucial to creating AI at the human level. We present an end-to-end general-purpose any-to-any MM-LLM system, to bridge the gap. We link an LLM to various diffusion decoders and multi-modal adapters so that MM-LLM can receive inputs and produce outputs of any type whether it be text, images, videos, or audio. By making use of the current highly effective and trained encoders and decoders.
Our MM-LLM is only tuned with a small percentage of certain projection layers' parameters (1%), which helps with convenient expansion to modalities that are more possible as well as low-cost training and the multiple use of different AI agents. Additionally, we introduce a modality-switching instruction tuning that enables complex cross-modal semantic comprehension and content creation for our MM-LLM. All in all, our project opens the door for more human-like AI research in the community by demonstrating the promising possibility of creating a versatile AI agent capable of utilizing different modalities simultaneously.

---

# Introduction:

Artificial intelligence-generated content has advanced to unprecedented levels recently in a variety of modalities. Text generation technologies like ChatGPT have demonstrated impressive language capabilities, and visual generation has been made possible by diffusion models. Large language models that demonstrate human-level language reasoning and decision-making, such as FLAN-T5, Vicuna, LLaMA, and Alpaca, have become increasingly popular. These models provide insight into how artificial general intelligence (AGI) is developing. Since humans perceive a variety of sensory data, including language, images, videos, and sounds, which frequently combine to create a multimodal world, text-based models are now being developed with knowledge of additional modalities, such as vision, video, audio, and more. These disparate but complimentary technological developments indicate that machines will eventually be able to interact with content in a rich, multisensory manner similar to how humans do.
Adapters are a promising method for aligning text-based large language models (LLMs) with pre-trained encoders from other modalities. Multimodal LLMs (MM-LLMs) such as BLIP-2, Flamingo, MiniGPT-4, Video-LLaMA, LLaVA, PandaGPT, and SpeechGPT have rapidly emerged because of this. Nevertheless, the majority of this research concentrates on comprehending multimodal content during information input, with the inability to produce content in more than text across various modalities. It is emphasized that smooth transitions between any information modality are a fundamental requirement of true human cognition and communication. To achieve true artificial general intelligence, or systems that can receive inputs in any format and respond in the appropriate modality, it is imperative to investigate MM-LLMs that can switch between any input and output modality.
To simulate human-like conversion between any two modalities, some attempts have been made. Though CoDi has recently made strides toward processing and generating arbitrary modality combinations concurrently, at its core it lacks the reasoning and decision-making capabilities of LLMs and is restricted to basic paired content generation. However, in order to achieve roughly "any-to-any" multimodal understanding and generation, efforts such as Visual-ChatGPT and HuggingFace GPT have attempted to combine LLMs with external tools. Unfortunately, because of their modular architectures, these systems have significant difficulties. First, discrete text outputs from the LLM are the only means by which information is transferred between modules. This means that errors and noise are inevitably propagated and introduced during the sequential process. More importantly, the entire system lacks end-to-end training on error and instead relies on pre-existing, trained tools for inference.
Consequently, content understanding and multi-modal generation capabilities may be severely constrained, particularly when it comes to deciphering intricate and implicit user instructions. In conclusion, the need to develop an end-to-end MM-LLM with arbitrary modalities is strong.
Our project, an any-to-any MM-LLM created to effortlessly handle inputs and outputs in any combination of four modalities—text, images, videos, and audio—aims to achieve this goal. Our project is divided into three parts, as shown in Figure 1. Firstly, we use well-known encoders to transcribe inputs from different modalities. A projection layer then projects these representations into language-like representations that the LLM can comprehend. Second, we process input data for semantic understanding and reasoning using an open-source LLM that is already in existence as the core. In addition to producing text tokens directly, the LLM also creates special "modality signal" tokens, which act as directives to specify which modality and whether the decoding layers should output content in accordance with them.
Since our project involves generating multiple modalities and encoding them, it would be expensive to train the system from scratch. Rather, we leverage pre-trained high-performance encoders and decoders that are already available, like ImageBind [25], Q-Former [43], and cutting-edge latent diffusion models [68, 69, 8, 2, 51, 33]. In addition to avoiding cold-start training, loading the off-the-shelf parameters also makes it easier for additional modalities to potentially develop. We only take into account locally fine-tuning the input projection and output projection layers for the feature alignment across the three tiers, with an encoding-side LLM-centric alignment and a decoding-side instruction-following alignment, where the lowest possible computational overhead guarantees improved efficiency.
Moreover, we introduce a modality-switching instruction tuning (called MSIT) that endows our any-to-any MM-LLM with human-level capabilities in complex cross-modal generation and reasoning, as well as sophisticated cross-modal semantic understanding and content generation. In order to address the lack of cross-modal instruction tuning data available in the community, we manually gather and annotate a high-quality dataset of 5,000 samples using MSIT. We optimize our project system overall using MSIT data by using the LoRA technique [32], which updates the LLM parameters and projection layers.
Overall, this work presents a promising avenue toward creating a more human-like MM-LLM agent that can represent universal modalities. The following are the contributions made by this project:
We hereby present our project, an end-to-end general-purpose any-to-any MM-LLM, which is capable of free input and output combinations of text, images, videos, and audio in addition to semantic understanding and reasoning.
We present two novel approaches to alignment learning: the instruction-following alignment at the decoding side and the LLM-centric alignment at the encoding side. These techniques effectively require very small parameter adjustments for successful semantic alignment.
To support MM-LLM in cross-modal content understanding and instruction reasoning, we annotate a high-quality modality-switching instruction tuning dataset that covers complex instructions across multiple modal combinations of text, images, videos, and audio.