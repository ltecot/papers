# Papers
Notes and summaries of papers. Not meant to be extremely accurate or thoughtful, so take it with a grain of salt. And along those lines, if you see something that is wrong, please open an issue or pull request!
"Reports" folder contains some of my work, mostly from independent research courses or other stuff that I want to archive but can't neccisarily be published.

### [Learning Symbolic Physics with Graph Networks](https://arxiv.org/abs/1909.05862)

Neat paper about learning physical laws (many-body gravity in this case) via graph neural networks. Seems that they basically create a structure where multi-layer perceptrons can operate on node pairs and update them per timestep. Then they use some previous work to try out algebraic equations that can approximate each MLP with minimal terms and error.

Seems cool, but I'd be interested in pushing this to its limits. What happens when it's a very complex equation, or if there's more complex interactions? Would be interesting to see if there's a sort of regret-minimizer equation finder in a setting where expirimental data is constantly fed in.

### [Certified Adversarial Robustness via Randomized Smoothing](https://arxiv.org/abs/1902.02918)

Basically we instead cerify robustness for a classifier that predicts the most likely output if we add gaussian noise to our base classifier. Although we have to approximate that smoothed classifier, so we can't certify with complete certainty, you can generally get better bounds with very high probability. I like this a lot, especially because it allows you to do away with all the linear programming nastiness of the formal verification procedures I'm used to (the cerification process is model-agnostic).

### [Vulnerability of quantum classification to adversarial perturbations](https://arxiv.org/pdf/1905.04286.pdf)

Some theorems about how adversarial robustness goes down as dimensonality increases in quantum information. I only really skimmed the theorems, but this seems sort of like an obvious thing. We see this happen too with classical machine learning. Interesting read, although at least from my current perspective doesn't seem like a particularly practical result.

### [The Learnability of Quantum States](https://www.scottaaronson.com/papers/occamprs.pdf)

Provides PAC bounds for learning quantum states. I only skimmed the proof but they basically seem to just apply a quantum mechanics hypothesis class to fat-shattering and concentration inequalities. Though the most interesting thing is they instead only bound the probability that the constructed quantum state will be close when it's measured in some dataset of specific measurments. (They say that previous work didn't allow this relaxation, and were only able to get an exponential sample size requirment, whereas they can get polynomial.)

Also, follow up work found [here](https://arxiv.org/pdf/1802.09025.pdf). I only skimmed it but basically seems to extend this into a regret-minimzation thing, where instead you're learning measurment after measurment and are just trying to minimize the amount you're incorrect over all measurments.

### [Towards Quantum Machine Learning with Tensor Networks](https://arxiv.org/pdf/1803.11537.pdf)

An overview of a method to do machine learning via quantum circuits (which can be equivalently represented with tensor networks). I got a little lost in some of the lingo but the general idea seems to be define a circuit with parameterized unitary gates, then optimize those gates via a numerical gradient (I think that's the right term, basically just try wiggling the parameters either way and go the better direction). 

### [A Quantum Approximate Optimization Algorithm for continuous problems](https://arxiv.org/abs/1902.00409)

Essentially just describing how to do optimization in a quantum algorithm. This is mainly an extension of Grover's algorithm, where you define an Hamiltonian that minimizes a defined cost function and evolve according to that (it's more involved than that but that's the TLDR). And then you can define other components to the cost to add in constraints, the same way you would do a Lagrangian in a regular machine learning method.

### [Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms](https://scirate.com/arxiv/1905.10876)

Essentially they just define metrics and do expiriments on the expressibility of different quantum circuits, meaning how many different states can they reach if you tune the parameters of the circuit. They also look at the level of enanglment you can achieve (in practice these metrics have precise definitions).

### [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175)

Very interesting paper that makes some empirical claims on adversarial examples. Essentially they run expiriments to show that models can generalize to noise that humans can't see, but are general features in the data itself. They do this through a few expiriments where they remove non-robust features from a dataset which makes the model robust with no extra tricks, and they show that models can perform well when only given these imperceptible features to work with (I didn't read the paper in detail so take what I saw with a grain of salt). Essentially the gist of what they are trying to argue is that adversarial examples are a result of features that humans can't precieve, but are there in a dataset.

I think this is pretty neat. I don't know exactly what these non-robust features tend to be, but I wonder if you could destroy them by just blurring or lowering resolution. If it hasn't been done already that'd be an interesting expiriment, see if you can improve robustness without hindering accuracy much by just reducing dimensionality. At the end they mention the need to encode human priors into the datasets, and I think I buy what they are selling. For instance, we take it for granted that rotating and image or changing small things shouldn't affect the output, but maybe that's not something that's generally true for all datasets. There's this whole question of how well should we expect our models to generalize, though I suppose it's not too unreasonable to say that a model shouldn't recognize an upside-down image unless it's seen upside-down images before. So I think that's what they mean by encoding human priors, you either need to somehow express that or just expand the dataset.

### [Model Based Planning with Energy-Based Models](https://drive.google.com/file/d/1XGYtcw4mX8zFwJmjkMPzJxbiIQm7Y5yV/view)

Energy based models for RL. I actually have a report where I tried stuff like this, though it didn't work nearly as well. To be honest what they exactly do isn't totally clear to me. But it seems like they do a more sophisticated version of what I did, which is essentially lowering the energy of the landscape while raising the energy of state transitions proportional to the reward they give you. I had all sorts of divergence and collapse issues when doing this, but presumably they've seemed to get it to work at least decently well.

### [Implicit Generation and Modeling with Energy-Based Models](https://arxiv.org/pdf/1903.08689.pdf)

I think I may have put this up here before, or a blog post link to it. But I like energy models so it's here anyways. Essentially you just have whatever model you want represent an energy function, and to get an output you minimize energy given an input. Then you just need to modify the weights to raise the energy of the examples you generate, and lower the energy of real examples. The specifics depend on exactly what type of model you want, and they have a lot of specific details here to get it to work well, but that's the gist.

They make a party platter of claims in this paper. I'm not really knowledgable enough to make any comments on them, with the exception of the robustness claims, which aren't particularly convincing to me. But irregardless still cool work to see.

### [Adversarial Policies: Attacking Deep Reinforcement Learning](https://arxiv.org/abs/1905.10615)

This is another paper I didn't actually read, but I ran into one of the authors at ICML and discussed it. Essentially train two agents in a two-player adversarial RL game, fix the policy of one agent, and then allow the other agent to search for adversarial examples by making small changes to their own policy at each time step.

I supsect if I read the paper in full there's probably more interesting analysis, but at a surface level this seems like a sort of obvious thing. If you fix the policy of one agent, unless you've come close to solving the min-max game you'll for sure be able to beat that agent. Though I do like this paper a lot, because it poses a very real question: is there a way for us to provide some sort of robustness (or guarantee of robustness) for adversarial agents when our own policy is fixed? It's a pretty realistic scenario for self-driving cars and such.

It seems like a pretty intractable problem in the unrestriced case, and I would guess that pretty much the best you could do there is just train more till nash equilibrium. But there are some assumptions we could make that might simplify it in a way that's still useful. For instance, we could instead just decompose this into a classification problem. Let's just assume there isn't an adversarial agent per say, and it's just someone tampering with the sensors or placing things in the enviroment that might cause failure. This would just decompose directly into standard adversarial guarantees and/or training, but instead of caring about classificaiton we care about action outputs. I'm not sure if that's neccisarily a useful study, because it's basically just copy and pasting research from one field to another. But presumably we'd care more about how these robustness guarantees translate to unseen states than we would in the classificaiton case, so maybe there'd be some interesting follow-ups there.

Also, say we restrict the adversarial agent to only small changes for very specific actions. This could correlate to non-uniform robustness guarantees, where we train to certify robustness in certain dimensions but don't care as much about it in others. I'm not really sure how you'd formalize this in a useful manner, but might be interesting.

The only other idea I'd have here is more of a game theory thing. Lets say we don't fix our own policy. Is there some way for us to best do defensive updates such that it's hard for the other agent to find these examples? To be honest I know zero about this field, honestly this might just be exactly the same as GAN training, I'm not sure it's really possible to do any better than just optimizing strictly for performance. But still, might be an interesting question to ask.

### [Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/pdf/1712.06567.pdf)

This is another skim, I didn't read the full paper. But I wanted to put it up here just because I'm a fan of this gradient-free stuff, especially in context of RL. To my understanding the title basically says it all, they just try using genertic algorithms instead of evolution strategies like OpenAI did.

### [Backpropamine: Training Self-Modifying Neural Networks With Differentiable Neuromodulated Plasticity](https://openreview.net/pdf?id=r1lrAiA5Ym)

Follow-up work from Uber on the differential-plasticity hebbian learning thing. I only skimmed it, but seems like they basically add other neurons in the network and/or a latent model that can "control" the amount that the hebbian value is passed through to the neurons. I was thinking something more along the lines of selectively preventing backprop on certain neurons and/or discouraging "far" neuron connections, though Jeff Clune talked about that stuff in his talk at ICML so I'm assuming those works are elsewhere?

### [Closed-Loop GAN for continual Learning](https://arxiv.org/abs/1811.01146)

I didn't actually read the paper in full, but I talked with the first author when she presented her poster at ICML. Essentially it's just a study on how a combination of a replay buffer and a trained GAN can help combat catastrophic forgetting. The only work I'm familiar with in catastrophic forgetting is the hebbian learning stuff Uber did, so I'm not really educated enough to give insight into this, but this seems pretty cool. I also really like their expiriment setup training MNIST for 0 and 1, then 2 and 3, etc. Very simple to understand and run on.

### [Matrices as Tensor Network Diagrams](https://www.math3ma.com/blog/matrices-as-tensor-network-diagrams)

Very nice intuitive blog post about representing matricies as tensor networks.

### [What physicists want to know about advances in generative modeling](https://machine19.github.io/machine19.github.io/Blogs/Albergo_GenModels/GenModels.html)

Not a paper, but a good overview of generative models by Michael Albergo. I like it because it has relatively simple explinations of everything. Unfortunately the part specific to physics topics isn't written yet, but hopefully that'll come soon.

### [Flow-based generative models for Markov chain Monte Carlo in lattice field theory](https://arxiv.org/abs/1904.12072)

Essentially they speed up the process of obtaining path integrals in a lattice via Monte Carlo chains by using flow models. To be honest, as is common with these physics papers, a lot of the finer details are over my head. But it's neat to see flow models used for these things, intuitively flow models seemed well-posed to me to do more of these probability-distribution related quantum computations.

### [A Survey of Quantum Learning Theory](https://arxiv.org/abs/1701.06806)

Pretty much what the title sounds like. They talk a little bit about different algorithms and the PAC learnability of a quantum state, but a lot of it is about learning the underlying function of a quantum oracle. Or at least I think, a lot of it is too over my head and I only skimmed so I'm not totally certain. I'm not totally clear on what purpose this has other than an academic curiosity, though I do know some applications in quantum chemistry provide you with a quantum state? So maybe there are instances of that sort of data that this would be useful for? I'd need to learn more to say for sure.

### [Quantum Machine Learning Algorithms: Read the Fine Print](https://scottaaronson.com/papers/qml.pdf)

Overview by Scott Aaronson about the caveates that come with different quantum learning algorithms. There are two things he covers. Firstly, a lot of the proposed algorithms require very specific properties to give speedups over classical ones. And secondly, a lot of them assume that we are given a specific input superposition. So in order for these speedups to still apply, it has to be at least equally hard to create the information classically as it is to put the information into a specific superposition (which apparently typically means a relatively uniform state). He also talks a lot about the HHL algorithm, which I believe is the algorithm that Ewin Tang found a classical counterpart for with comprable complexity.

### [Integrating Neural Networks with a Quantum Simulator for State Reconstruction](https://arxiv.org/abs/1904.08441)

Similar to the paper below, but a more high level summary.

### [Machine learning quantum states in the NISQ era](https://arxiv.org/abs/1905.04312)

An overview paper of how machine learning is being used to learn quantum states. Essentially they take a bunch of measurments and then create a generative model that learns to draw measurments the same way the underlying state does. They mostly use Boltzmann machines, and I've been told that they mainly do that because they work decently well and physicists like them (though other generative models would work fine too presumably).

I like this sort of stuff because it's sort-of the empirical side of what's the boundary between classical and quantum models of computation. I'm curious about how that boundary applies to learning different operations on quantum states.

### [TensorNetwork: A Library for Physics and Machine Learning](https://arxiv.org/abs/1905.01330)

Essentially documentation on Google's tensor network's library.

### [Qubit Allocation for Noisy Intermediate-Scale Quantum Computers](https://arxiv.org/abs/1810.08291)

Paper about quantum algorithm compiling. Basically it's a standard constraint satisfaction problem where you have to assign logical qubits in the algorithm to qubits on your specific machine's architecture, while minimizing the number of swaps that have to be performed to conform to the qubit's connectivity. They use standard search algorithms and simulated annealing.

### [Learning Logistic Circuits](https://arxiv.org/abs/1902.10798)

Basically about combining symbolic AI with more modern approaches. Take what I say with a grain of salt cause I mostly skimmed it, but it seems that essentially they create circuits that encode some sort of symbolic knowledge about the system. You can give each input a certain probability and then compute the probability of the output at each gate, which they call the "circuit flow". Then they define a weight function with parameters that are tied to the "circuit flow", and you can learn these parameters via normal gradient descent. Pretty neat, not something I really thought of before, but the more I think about it it's kinda what people have been doing all along on a more abstract level. We choose different network structures for different jobs because we know the data tends to follow certain patterns. This is just a much more formal way of doing that via symbolic logic.

### [Implicit Generation and Generalization in Energy-Based Models](https://openai.com/blog/energy-based-models/)

OpenAI blog and paper about energy based models. It is essentially the same process as equilibrium propagation, though I think this is generally true for all energy models. Basically you give an input, minimize the energy from that initial state with respect to the input, then update the weight parameters by contrasting the produced input with the ground truth input. The difference here is that the energy function is the network itself, instead of something we directly define. Interesting stuff, they do a lot of expiriments to show it doing different things. Because defining it as an energy function makes it sort-of flexible they can do things like generate samples and such.

I haven't really done a lot of stuff with these types of models but I think you could do the same type of thing with just regular networks. For example, for the generative stuff you'd just have to define a loss that maximizes your desired output, then do gradient descent with respect to the inputs. I am sort of curious though what affects this training procedure has versus doing just plain squared error loss training. They mention some stuff about it having better adversarial robustness, which I could see being true given that the training procedure essentially trains against self-generated counterexamples. Though I'm not really familiar enough with the topic to say if that's actually a procedure that works with current neural networks.

To those confused as to how this type of process of sampling can approximate the entire energy landscape, the intuition is pretty well explained by [Hinton's paper](https://onlinelibrary.wiley.com/doi/epdf/10.1207/s15516709cog0000_76) that they cite. Basically if you do a walk to minimize energy, in theory the vector with the minimal energy will dominate the landscape. So if you keep sampling via minimzing energy you should be sampling roughly proportional to what you would get by just pulling it directly from the probability distribution.

### [The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation](https://arxiv.org/pdf/1802.07228.pdf)

A lengthy report on the many aspects of dangers of intelligent systems. Put together by Miles Brundage and many other authors from various institutions. Definitely recommend to read, especially if there are specific areas you're interested in you can just skip to them.

Personally I think the best take-away from this article isn't neccisarily the extistential stuff with super-intelligent AI, because I think that's a generally understood argument. I think (at least from my current perspective) what is less understood is the impact of even modern-level agents. For instance, when Dr. Russell was doing his visiting lecture, he talked about how social media has been an example of this. The algorithms these websites put in place to increase the number of click-throughs seems like a pretty innocent and reasonable thing at first. But eventually it leads to forms of addiction, polarization on the political spectrums, proliferation of false news, etc. (in theory, I'd want to see a study if I were to make definitive claims). There are numerous examples of more near-term concerns similar to this.

### [The Off-Switch Game](https://arxiv.org/abs/1611.08219)

I came across this paper because of a talk Dr. Russell gave at UCLA the other day. It made me think of alignment more as a property that should be heavily ingrained in the way we design agents, rather than a tack-on to a reward function. This paper was an example he used, specifically in how acting on the percieved desires of a human, and being uncertain about said desires, actually gives incentive for a theoretical artificial agent to allow itself to be turned off. (Whereas if it operates off pure reward, even if being turned off isn't explicitly labelled as bad, it can be interpreted as bad because being turned off means it can't gain any more reward.)

To quote the author of the paper, "A rational human switches off the robot if and only if that improves the humans’s utility, so the robot, whose goal is to maximize the human’s utility, is happy to be switched off by the human." The proof is more involved but that's essentially the summary of how this type of safety protocol works. However there are two assumptions critical to the proof. One is that the robot is uncertain about the human's utility, and the other is that the human is semi-rational (meaning non-random preferences for deciding whether or not to switch the robot off given an action request).

The paper starts out by essentially deriving the delta between asking the human vs. just taking an action. They prove that this delta is always positive if the human has rational preferences (IE probabilities of 1 and 0 for specific actions). To deal with this they introduce a different belief equation (equation 5) that assumes the human will make irrational choices proportional to the utility of each action. (AKA, if the action is clearly very good or very bad they probably won't mess up the choice, but if it's of little consequence they might give a somewhat random answer.)

Overall very interesting work. It's still all theory and there are a lot of open problems around this. For instance, the authors mention in remark 2 that it's important to accurately represent uncertainty in their models in order for this to work. But it's a very nice work to outline the importance of uncertainty in safety, which isn't something I really considered before now.

### [Spurious Local Minima are Common in Two-Layer ReLU Neural Networks](https://arxiv.org/pdf/1712.08968.pdf)

Interesting paper that was presented in a graduate seminar class I'm taking. Essentially they create a target and trained neural network, and the goal is for the trained network to learn the function of the target. They show that for a few specific configurations, when both networks have the same number of neurons there are many spurious local minima in the loss landscape. However, if you add an extra neuron or two per layer, there are significantly less local minima. This doesn't neccisarily prove anything concrete in the general case, but it provides some interesting insight. The idea being that if you have a neural network that only a little more expressive than your true underlying data distribution, then in theory it should be able to learn it well despite it being a non-convex learning problem. Once again this is only really an indication, it's a little difficult to think about what is the complexity of the "data distribution" for data like video games or images, plus this isn't a general proof. But it's definitely an interesting result to think about and build on.

### [Unsupervised Learning via Meta-Learning](https://openreview.net/pdf?id=r1My6sR9tX)

Paper in review on meta-learning for unsupervised learning. Take my summary of this with a grain of salt because I skimmed pretty heavily. But, it seems that they take embeddings of inputs, cluster these embeddings to automatically create supervised tasks, and then run a meta-learning algorithm on this process a bunch of times (figure 1 illustrates this pretty clearly). Seems to be a pretty standard furthering of messing around with meta-learning, which I'm a fan of. Though personally I think I'd be more interested in the embedding function. Though I'm not too familiar with the literature space, so I'll have to read more into the stuff they used in this specific paper. But I am a fan of the idea that generalized learning is about learning how to strucutre your data such that it becomes easy to perform tasks that give high reward.

### [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)

Survey on different aspects and approaches to AI safety. Good read for an overview and some ideas.

### [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)

Paper from Nvidia that produces extremely lifelike generations of human faces. I need to read this in detail but I wanted to place this here because the results are super neat.

(TODO: Return to this paper.)

### [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)

About improving the ability to learn environment dynamics. In my opinion this paper is a little dense and jargony, but there's a nice summary that helped me understand [here](http://www.shortscience.org/paper?bibtexKey=journals/corr/1811.04551#wassname). To basically restate what they said, the main things they do here are train on embedded features (not neccisarily straight on the pixels), use both deterministic and probablistic information in a recurrent model, and train to look many steps ahead. If you look at the functions provided in Equation 1, this makes more sense. Essentially they have a variety of models they train, and then as demonstrated in Algorithm 2, they sample action trajectories and re-fit their models to the best of these trajectories.

This probably also deserves a closer look at a later time.

### [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

Interesting paper that won best paper at NeurIPS 2018. To be honest I lack the knowledge to understand a lot of this, so I'll have to come back and give it a better read when I have more time. Until then I can't give any sort of informed opinion on it.

(TODO: Return to this paper.)

### [Go-Explore](https://eng.uber.com/go-explore/)

Post from Uber research that claims massively improving Montezuma’s Revenge and Pitfall scores. They still have yet to release the paper though. The general idea is you remember states you haven't explored, return to them, and keep exploring. Basically this is meant to ensure the algorithm doesn't give up on going down a path it previously tried. The results are flashy, and it's definitely a nice intuition I've seen reflected in other papers too. Though there are issues here with practicality, namely in that the results assume determinsitic environments for the most part. A good discussion of pitfalls is discussed [here](https://www.alexirpan.com/2018/11/27/go-explore.html?fbclid=IwAR315UVwD1503QMba-7Y3BoLulWlxCxR6zHEclLrXk5FuinEblF_4R5CReQ).

### [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://medium.com/@mayban/a6668993bd8)

Implementation details on Waymo's machine learning pipeline for car control. I only skimmed, but seems interesting. Always a fan of companies, especially ones at the scale of Waymo, publishing how they do their work. Though it did make me think about simulated examples. It'd be interesting to see some form of GAN for this, where scenarios are generated with the intent of trying to make the system fail.

### [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179)

This paper is the semi-culmination of the algorithmic work done at MILA with Bengio on local plasticity, hebbian learning, whatever you want to call it. There's a few follow up papers ([here](https://arxiv.org/abs/1711.08416) and [here](https://arxiv.org/abs/1808.04873)) that talk about and prove properties of this algorithm. Though for the most part it's pretty similar to the contrastive hebbian learning paper, except that the outputs aren't fully clamped to the desired output. The majority of this paper just goes deep into the theory, implementation, how it differs and is similar to previous work, details of how to get it to work and hypothesis for future findings, etc. Solid read if this is an interesting method to you.

Personally I'm interested in making this architecture and optimization inherently tied to reinforcment learning. I believe the idea behind the two stages of the training process are to increase the energy of undesirable states, and reduce the energy of desired states. So when we get an input and run the network to minimize the energy, it'll converge to the desired outputs. But on an intuitive level it seems to be pretty easy to formulate it in the context of reward, because you could just decrease the energy for high reward and increase it for low reward. (Lots of practical and theoretical problems to address for this but it's an idea I'd like to explore.)

### [Supervising strong learners by amplifying weak experts](https://arxiv.org/abs/1810.08575)
![](https://github.com/ltecot/papers/blob/master/amplify.png)

This paper essentially suggests a way to interatively improve on a learner through human help. The diagram above from the paper does a pretty good job of summing up what's going on. Basically you have a human or some teacher give answers to a question, and it can ask the learner smaller sub-questions that it might need to solve the question. The algorithm then learns to imitate the teacher for that question, and it then throws the ability to answer that question into X. They talk about desirable qualities related to avoiding reward functions that could cause the learner to do very catastrophic things (doomsday AI stuff), and also better efficiency through reusing previously learned stuff structurally and whatnot.

I didn't read too much into the specific implementation details, but this kind of concept is something I've been seeing around that I like. There's sort of an implicit heirarchy here. We learn how to do a bunch of things, and then we can learn to do more things by decomposing a new things into things we know. And it's an evolving structure as we encounter new tasks, so the heirarchy is not fixed. Incorperating this type of evolution into a learner that doesn't need to take in human examples would be neat to see, like some form of fluid heirarchy in an RL algorithm. Thought I'm sure theres a paper out there that trys to do this.

### [A Universal Training Algorithm for Quantum Deep Learning](https://arxiv.org/pdf/1806.09729.pdf)
![](https://github.com/ltecot/papers/blob/master/phase_kick.png)

There is a [youtube video](https://www.youtube.com/watch?time_continue=6&v=uxL-wbuvpj0) where the author goes over the core concepts. Personally I've only watched this video, the paper is like a mini book on the topic. There is some really cool stuff here in how to how train modern versions of neural networks fully in a quantum computer.

Honestly a lot of the mathematical and quantum computation topics are beyond my current understanding (so take what I say with a grain of salt), but the idea illustrated in the figure above from the paper sums up a key concept about how this all works at a conceptual level. Essentially using a notion of loss, you can "kick" the distribution of the input's quantum state to minimize that loss. You can think of it as you take all possible inputs and kick them to a more optimal state all at once. Then when you finally measure the state at the end of the process, you will have a high chance of sampling a very good solution. It goes along the general intuition that a quantum system allows you to consider all possibilities at once (if I understand it right). (By the way, the top part of the diagram shows the process if you had a checkpoint and measured the value in the middle of the optimization process.)

Really cool paper and he implemented a lot of this work on example problems with Rigetti cloud resources. One thing I thought of though (and he addressed this too) is about these ideas influencing classical ML and vice versa. [Ewin Tang](https://ewintang.com/) has been starting a spree of disproving quantum superiority in [recommender systems](https://arxiv.org/abs/1807.04271) and [PCA](https://arxiv.org/abs/1811.00414), so if this is any indication of a trend it'll be cool to see what this type of work inspires in studying different forms of learning algorithms.

### [Towards deep learning with spiking neurons in energy based models with contrastive Hebbian plasticity](https://arxiv.org/pdf/1612.03214.pdf)

An implementation of networks optimized through hebbian-like local plasticity rules. They essentially formulate the learning problem as loss constrained by an energy equation, and then solve it through a relatively simple update rule. Seems promising and interesting, though to be honest I don't understand some of the math so I'll need to spend a little longer with this. The algorithm steps on page 4 break it down in very simple terms. There's essentially a feed-forward where you subtract from weights that have any correlation, and then a feed-backwards where you fix the output, let the energy minimize for a while and then add any correlations. You do this by essentially continually minimizing the energy equation and alternating the weights that correspond to how important the difference in output is between 0 and some positive value.

This is really interesting but the intuition is a little lost on me. The paper this work is built off of the paper [Towards Biologically Plausible Deep Learning](https://arxiv.org/pdf/1502.04156.pdf), which has a really nice diagram on the third page that illustrates how synaptic learning works in theory. In essence, you want to decrease correlation between a connection that causes another unit to fire, but if there is simple correlation and not temporal causation, you want to decrease those weights. (Or at least that's my general understanding, I didn't read this paper in full yet.) I'd like to see these ideas applied to a variety of other areas. Looking at optimization for this type of learning, maybe simpler rules, applying to RL, etc.

### [Differentiable plasticity: training plastic neural networks with backpropagation](https://eng.uber.com/differentiable-plasticity/)

This blog post and the attached paper basically specify a form of learning that combines the hebbian learning concept with traditional deep learning practices. Basically they train the network with two parameters: typical activation weights, and a term for the weight of the hebbian value, which is essentially a running average of the tendency for both neurons to be activated at the same time (they use a little extra trick to prevent decay of the hebbian value of an inactive neuron over long time horizons). They train the weights with typical SGD methods, and then the hebbian parameter is a more plastic term to allow better expressibility and memory. And it seems to work well in the contexts of the tasks they tried out.

There are some interesting ideas here. Firstly, the idea of combining traditional static weights with backprop with a more plastic learning method. Their anecodtal justification for this is saying that this initial learning is like the equivalent of going through natural selection to get the human brain, and that the hebbian part is the learning over a lifetime. I can't really speak on that, but I think it's a nice trick to help bootstrap new optimization methods. Using hebbian and local-plasticity methods right now isn't as tried and tested as SGD, and I'd guess that a lot of the more highly supervised problems won't really benefit from it. So this setup likely makes it easier to get to a point where you can observe interesting behaviors and performance improvments. You could in theory place any sort of hebbian rule you like into this model, though I'm not sure how stable it will be if you throw something like what was used in the [contrastive Hebbian plasticity](https://arxiv.org/pdf/1612.03214.pdf) paper. In theory it's just a method of constraining the hypothesis space, so as long as you know that your early training is representative of what the model will mostly be dealing with I'm ok with that.

They mention papers like [MAML](https://arxiv.org/abs/1703.03400) and [SNAIL](https://arxiv.org/pdf/1707.03141.pdf) papers, saying that they got similar performance with much fewer parameters. I haven't read either in great detail, but the general gist I get with MAML is that they train over a wide variety of tasks, and with SNAIL they add a bunch of layers with convolutions through time and attention modules to create more expresibility in terms of accessing previous inputs. Although pushing this stuff is important, I do like the author's implied point here. I think many of these problems would benefit from exploring other forms of optimization + networks and understanding which ones create which behaviors. Not just seeing what we can do by throwing more data and more layers + different types of expressability at our models, but investigating what rules allow desired behaviors to be emergent.

Though I will say that I'm not totally sold on their hebbian learning equation. In the context of memory I can certaintly see how it would be useful, because it encodes patterns that have been seen before by increasing the probability of neurons "firing" if they have fired together before in the past. But in the context of learning, it's essentially just a complex term to increase the expressibility of the model. There's a lot in the area this type of learning dealing with hebbian local plasticity updates (change in update direction based on post and pre synaptic firings, dynamic weights within the hebbain learning model, constraints for computational efficiency) that would be nice to explore here. Also, I'm not really a fan of how they don't really give any theoretical justification for their method, and they kinda waive around claims. But that's more of a research-level nitpick from me. I'm certaintly a fan of having both a healthy dose of the theoretical and empirical, so hopefully the community as a whole around these concepts builds on this.

### [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/pdf/1806.10293.pdf)

Best paper winner at CoRL. The TLDR is they got really good at picking things up with robot arms. It's important work to push the boundaries and build practical systems out of what works. But on the RL side, I think the algorithm they use is of the most interest. It's a Q-learning algorithm, which typically I haven't seen in recent publications in favor of actor-critic methods.

It's actually pretty simple. They are in a sense still doing actor-critic. Except instead of value estimator, they are using a Q-value estimator, and instead of an actor NN, they are just fitting a gaussian using cross-entropy to estimate the action that produces the highest value from the Q-values given a state. It's the same pipeline, except they're removing a lot of the weirdness of optimizing a gradient on top of another changing network. At least in theory, from a thought-expiriment and empirical point of view. This is pretty cool, I think I'll try this out on some atari games and see if it translates nicely.

### [Reinforcement Learning with Prediction-Based Rewards](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/)

This blog post is essentially the combined work of two papers: [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/pdf/1808.04355.pdf) and the work built on it by [Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf). The general crux of it is that if the agent can accurately predict what will happen, it will avoid those scenarios in favor of states that it doesn't know. This is formalized by adding the difference between observed and predicted states as a reward.

I like this a lot, because it's driving towards the idea of learning in general rather than learning for a reward. It'd be cool to see this work continued in the context of curiosity guided by reward (maybe multiply estimated values with error of prediction values for reward?).

### [Differentiable neural computers](https://deepmind.com/blog/differentiable-neural-computers/)

I only skimmed the blog, (and in ML community standards this paper is a little old) but I wanted to write a quick thing about this because I really like it. I'd expect there's also a lot of new stuff out there around this paper that I should read.

The TLDR here is essentially you make a policy that decides operations to take given a set of operations as actions and a set of memory it has access to to write and read. I like thinking about different rules of operation and optimization for neural networks that provide better memory and modeling of the environment, but I think the idea of giving it access to a memory bank is a cool idea. It sort of implicitly gives the model a lot of memory to work with. And you can fit this model into typical reinforcment learning setups. I'm not sure with what this has been proven to work with yet, but I'd like to expiriment with this more.

### [Reinforcement Learning of Physical Skills from Videos](https://arxiv.org/pdf/1810.03599.pdf)
Really cool paper on learning different skills, such as flips, kicks, dances, etc. from video. They essentially built a system to extract a nice 3D model of the action in the video, and then they ran a RL algorithm that got reward based on how closely it could match that model over time. And they were able to train + augment it to work with varying interferences in the environment, plus with different body types. I'd love to see essentially just pushing this further. Maybe see if you can fit it to extremely different body types (spider-type, dog-type). Or just cool stuff, like throwing the trained models into a video game and allow someone to give high-level controls and have the agent follow them in a physically simulated way.

### [Shallow Learning For Deep Networks](https://openreview.net/pdf?id=r1Gsk3R9Fm)
![](https://github.com/ltecot/papers/blob/master/shallow_learning.png)

An iterative process of training neural networks (this this case, a CNN). The picture above from their paper is pretty illustraive of how it works. They essentially train one layer at the time. I didn't read very deeply into their theorems and details, but the general theme seems to be that this allows them to have the nice training properties of single layer neural networks, because they can treat the previous layer just as a regular input. And this allows them to get pretty good results with a smaller network. This also continues my interest on seeing if there can be some form of general network, where neurons and their architecture aren't constrained and are learned / altered through time. And also if they can be trained to represent useful concepts locally, in order to account for a vastly changing goal.

### [Capacity and Trainability in Recurrent Neural Networks](https://arxiv.org/pdf/1611.09913.pdf)
Overview of a bunch of different RNN architectures. They do a variety of analysis with hyperparameters, number of parameters, number of units, etc. Most of the analysis centers around the network's ability to reproduce binary input it saw in the past, and they use that get the bits per parameter measure of the network. They essentially found that most networks get 3-6 bits per parameter, though strictly speaking this measure might not be particularly useful because the LSTM and GRU networks that had worse memory per parameter train better on more practical tasks. And a little interesting tidbit I liked is they say bioligcal synapses were calulated to have similar capacity, 4.7 bits. Although the this paper is more focused around analysing RNN architectures for choosing and training current models, I'd be interested in seeing similar analysis in RNN's ability to encode concepts, causal relations, etc.

### [Learning Hierarchical Information Flow with Recurrent Neural Modules](https://arxiv.org/pdf/1706.05744.pdf)
![](https://github.com/ltecot/papers/blob/master/ThalNet_diagram.png)

I think this picture from the paper is essentially all you need to get the gist of the idea. There's definitely more to the implementation of the model, but personally I didn't find that or the results particularly interesting. It's the concept of having a heirarchy or recurrent units that I found interesting, and they were inspired to try this out by the thalamus. I would love to see work around this on seeing if you can observe this type of compartementalization emerge training recurrent networks that don't force the constraint. Perhaps something around neural spiking, or using some form of other penalization than encourages heirarchy.

### [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://blog.openai.com/evolution-strategies/)
Essentially just trying out a bunch of slightly different policies by slightly changing the parameters, choose the best one, rinse and repeat. And this can be faster just because when you're working with a hidden loss like reward, sometimes it's just faster to try it out than computing the gradient. I can see this being a problem with enviroments that involve a lot of other agents or it takes a very long time to play out. Though I only looked at the blog, not the paper and code, so they may have addressed concerns like these more directly there. I could also see this being incrementally improved by doing something like baising the sampling of new parameters in the direction of the previous updates, to make the sampling a little smarter.

### [Meta Learning Shared Hierarchies](https://blog.openai.com/learning-a-hierarchy/)
Essentially just throw RL algorithms on top of eachother. The highest one runs at a longer timestep. It chooses which lower level algorithm to use, and then it's "reward" is just the sum of whatever the lower level model achieved in that time. Pretty cool, at least in the examples they gave seemed to learn good seperations of tasks. I'd like to see someone find counterexamples, or to generaize the system of layers. Essentially "how deep do you go?".

### [Model-Based Reinforcement Learning via Meta-Policy Optimization](https://arxiv.org/abs/1809.05214)
This paper continues the push for what Abbeel calls meta-learning. Specifically it builds off [
Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) by Finn, Abbeel, and Levine. In that paper, they suggest instead of training a RL model on just one task, to update parameters according to rewards from a wide variety of tasks, and then fine-tune the weights during "test" time for each task initialzied from the global optimal. It's a smart way of pushing the power of NN's another step further, and you can draw parallels given the success of transfer learning in vision and how biological brains "train" over all "environments" in their life (in theory at least).

They've pushed this idea further before with [evolved policy gradients](https://blog.openai.com/evolved-policy-gradients/), which probably deserves a section of it's own here. The idea being that loss functions are more general and share more similarities between differing tasks, so it'd be best to instead optimize over all environments on a new loss function rather than directly on the policy. The blog post explains this well, but the policy and loss model essentially are a form of cooperative network. The policy uses the reward function to try and learn an enviroment as it normally would, it returns the reward it got, and then the reward function uses that value to do gradient descent on it's own parameters.

And finally, this paper goes in another direction of meta-learning. Instead of having an ensemble of policies, they have an ensemble of models. The algorithm flow in the paper illustrates the process pretty well. First, they sample trajectories from the real enviroment. Then, using randomly sampled steps from their entire real-life history, they train *K* models that estimate the next state, given action *a* and current state *s*. These models are meant to represent *K* different possible MDPs of the true enviroment. Then, they gather simulated data from each of these possible MDPs. Each of these estimated MDPs has it's own policy model (in this paper they use TRPO). For each *K* models, they will sample the corresponding estimated MDP, update the model's parameters using gradient descent, and sample new simulated trajectories using this updated policy. Then, using this new simulated data and the algorithms from the first paper (called MAML), they update the meta-policy using gradient descent. Finally, they run more trajectores in the real world using each *K* policy, rinse and repeat till it works well.

That's a lot of words, but the gist of it is that there's *K* models that form their own ideas about how the enviroment acts, and how to get the best reward given that model of the world. They get that idea by observing real data, and by running simulations on their own model to determine the best policy. The meta (or global, top-level, whatever-you-want-to-call-it) policy is then updated using even more simulated data from the possible models and policies.

There's been work done before like this on combining model-based RL and model-free RL to improve performance (I can't find the link after a quick search but David Silver talks about it in his lectures). It seems that this is essentially just applying the meta-paradime to it. Though this paper also seems to assume you know the reward function, though in theory you could just add that to the estimated MDP to learn from the real data. I would like to see more analysis on how the meta aspect of this algorithm improves performance. Does each estimated MDP actually internalize different concepts if you give each randomly sampled data? Or is it just a way to get more diverse exploration.

### [A Semantic Loss Function for Deep Learning Under Weak Supervision](https://web.cs.ucla.edu/~guyvdb/papers/XuLLD17.pdf)

Essentially they constrained a NN to also include loss from breaking logical sentences, presumably that decribe the rules of the task in a logical way (I didn't read too deep). Good really straightforward example of why this type of reasoning is important to try to include in modern methods. They didn't neccisarily see better results, but they were able to achieve optimal solutions faster and with less data.
