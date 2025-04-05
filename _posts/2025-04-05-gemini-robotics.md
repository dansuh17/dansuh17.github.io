---
layout: post
title: "Gemini Robotics"
date: 2025-04-05 12:59:00 +0900
---

Notes after reviewing [Gemini Robotics technical report](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf).

2 years after introducing [RT-2](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/), Google DeepMind revealed another breakthrough in robotics foundation model: [Gemini Robotics](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/).
Gemini Robotics demonstrated enhanced capabilites compared to RT-2 - better generalizability, planning capabilities, and general embodied reasoning.

<!--more-->

# Gemini Robotics - Embodied Reasoning (ER)

Gemini Robotics-ER is an advanced vision-language model (VLM) that enhances Gemini's spatial reasoning capabilities that are necessary for robotics.

* **Architecture**: Gemini Robotics-ER uses Gemini 2.0 Flash as backbone and is fine-tuned to understand spatial reasoning. However, in the technical report it is not clear how exactly it acquired better spatial reasoning capabilities.

* **Enhanced spatial reasoning**: The authors emphasize four different categories of improved out-of-the-box embodied reasoning:
  1. Object Detection: Identifying objects in images through 2D bounding boxes. Represented as $(y_0, x_0, y_1, x_1)$ quadruples.
  2. Pointing: Identifying points in images specific objects or object parts. Represented as $(y, x)$ tuples.
  3. Trajectory Prediction: Predicting object or action trajectories given a description of the motion. Represented as a sequence of points connecting two points.
  4. Grasp Prediction: Predicting top-down grasps (i.e. where to grab at what angle). Represented as $(y, x, \theta)$ where $\theta$ is the rotation angle.

* **Connecting embodied reasoning to robot control**: Using Gemini Robotics-ER's exceptional spatial reasoning capabilities, it is possible to control robots in a zero-shot or few-shot manner. 
  _Zero-shot control_ leverages Gemini 2.0's innate language capabilities and uses code API generation. Given an image, prompt, and the API specification for the robot, it can generate it's plan and a sequence of API calls to achieve the task without ever being fine-tuned. Gemini Robotics-ER can perform more dexterous tasks through _few-shot control_ where it is presented a high-quality reference data.

# Gemini Robotics

Gemini Robotics is described as an advanced vision-language-action model (VLA) that is built upon Gemini Robotics-ER, where the key difference is that it can directly output actions, similar to RT-2.

* **Architecture**: Gemini Robotics is Gemini Robotics-ER fine-tuned to directly output actions. However, the technical report omits how exactly the actions look like. One can presume that it might look similar to the prior work, RT-2, where there is a predefined set of tokens representing robotic actions. The system takes the cloud backbone + local decoder approach, where only the small action decoder is present in the robot due to sizing & latency constraints. However, it is unclear what data is being communicated between the backbone and the local model.

* **Generality**: With the ability to directly output actions, Gemini Robotics has the capability to perform various robotic tasks requiring planning and dexterity. Such tasks include folding laundry, tacking plates, opening a folder, and picking up a shoe lace. Also, it achieves remarkable generality in three areas: visual (unseen background or lighting conditions), instruction (paraphrasing and typos), and action (different object placement, color, or shape). The authors present $\pi_0$ reimplemented and multi-task diffusion policy as baselines, and Gemini Robotics outperforms generalization benchmarks.

* **Specialization**: Gemini Robotics can be specialized to perform more dexterous tasks with long-horizon planning. It can also be specialized to adapt to different robot types (multi-embodiment). This can be done by fine-tuning the model with a smaller set of high-qality action data demonstrating complicated tasks. As a result, it can perform impressively complicated tasks requiring high level of dexterity such as [packing a lunch box](https://www.youtube.com/watch?v=m-G4-slYcGE), [making a salad](https://www.youtube.com/watch?v=1r2GqaGyyIA), and [playing cards](https://www.youtube.com/watch?v=1Dkdyrt6bt0).

# Limitations

Here are the areas where Gemini Robotics falls short or shows room for improvements.

* Highly dexterous task such as inserting shoe laces.
* "Grounding spatial relationships across long videos" (== very short memory)
* Numerical precision in pointing / object detection.
* Zero-shot cross-embodiment transfer.

# References

* [RT-1 project page](https://robotics-transformer1.github.io/)
* [RT-2 paper](https://robotics-transformer2.github.io/assets/rt2.pdf)
* [RT-2 project page](https://robotics-transformer2.github.io/)
* [pi0 paper](https://www.physicalintelligence.company/download/pi0.pdf)
* [pi0 blog post](https://www.physicalintelligence.company/blog/pi0)
* [U of T Robotics Institute Seminar: Sergey Levine (UC Berkeley)](https://www.youtube.com/watch?v=EYLdC3a0NHw)
