# Papers
Notes and summaries of papers for myself. Not meant to be extremely accurate or thoughtful, so take it with a grain of salt. And along those lines, if you see something that is wrong, please open an issue or pull request!

## Reinforcment Learning

### [Model-Based Reinforcement Learning via Meta-Policy Optimization](https://arxiv.org/abs/1809.05214)
This paper continues the push for what Abbeel calls meta-learning. Specifically it builds off [
Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) by Finn, Abbeel, and Levine. In that paper, they suggest instead of training a RL model on just one task, to update parameters according to rewards from a wide variety of tasks, and then fine-tune the weights during "test" time for each task initialzied from the global optimal. It's a smart way of pushing the power of NN's another step further, and you can draw parallels given the success of transfer learning in vision and how biological brains "train" over all "environments" in their life (in theory at least).

They've pushed this idea further before with [evolved policy gradients](https://blog.openai.com/evolved-policy-gradients/), which probably deserves a section of it's own here. The idea being that loss functions are more general and share more similarities between differing tasks, so it'd be best to instead optimize over all environments on a new loss function rather than directly on the policy. The blog post explains this well, but the policy and loss model essentially are a form of cooperative network. The policy uses the reward function to try and learn an enviroment as it normally would, it returns the reward it got, and then the reward function uses that value to do gradient descent on it's own parameters.

And finally, this paper goes in another direction of meta-learning. Instead of having an ensemble of policies, they have an ensemble of models. The algorithm flow in the paper illustrates the process pretty well. First, they sample trajectories from the real enviroment. Then, using randomly sampled steps from their entire real-life history, they train *K* models that estimate the next state, given action *a* and current state *s*. These models are meant to represent *K* different possible MDPs of the true enviroment. Then, they gather simulated data from each of these possible MDPs. Each of these estimated MDPs has it's own policy model (in this paper they use TRPO). For each *K* models, they will sample the corresponding estimated MDP, update the model's parameters using gradient descent, and sample new simulated trajectories using this updated policy. Then, using this new simulated data and the algorithms from the first paper (called MAML), they update the meta-policy using gradient descent. Finally, they run more trajectores in the real world using each *K* policy, rinse and repeat till it works well.

That's a lot of words, but the gist of it is that there's *K* models that form their own ideas about how the enviroment acts, and how to get the best reward given that model of the world. They get that idea by observing real data, and by running simulations on their own model to determine the best policy. The meta (or global, top-level, whatever-you-want-to-call-it) policy is then updated using even more simulated data from the possible models and policies.

There's been work done before like this on combining model-based RL and model-free RL to improve performance (I can't find the link after a quick search but David Silver talks about it in his lectures). It seems that this is essentially just applying the meta-paradime to it. Though this paper also seems to assume you know the reward function, though in theory you could just add that to the estimated MDP to learn from the real data. I would like to see more analysis on how the meta aspect of this algorithm improves performance. Does each estimated MDP actually internalize different concepts if you give each randomly sampled data? Or is it just a way to get more diverse exploration.

### [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://blog.openai.com/evolution-strategies/)
Essentially just trying out a bunch of slightly different policies by slightly changing the parameters, choose the best one, rinse and repeat. And this can be faster just because when you're working with a hidden loss like reward, sometimes it's just faster to try it out than computing the gradient. I can see this being a problem with enviroments that involve a lot of other agents or it takes a very long time to play out. Though I only looked at the blog, not the paper and code, so they may have addressed concerns like these more directly there. I could also see this being incrementally improved by doing something like baising the sampling of new parameters in the direction of the previous updates, to make the sampling a little smarter.

### [Meta Learning Shared Hierarchies](https://blog.openai.com/learning-a-hierarchy/)
Essentially just throw RL algorithms on top of eachother. The highest one runs at a longer timestep. It chooses which lower level algorithm to use, and then it's "reward" is just the sum of whatever the lower level model achieved in that time. Pretty cool, at least in the examples they gave seemed to learn good seperations of tasks. I'd like to see someone find counterexamples, or to generaize the system of layers. Essentially "how deep do you go?".

### [Reinforcement Learning of Physical Skills from Videos](https://arxiv.org/pdf/1810.03599.pdf)
Really cool paper on learning different skills, such as flips, kicks, dances, etc. from video. They essentially built a system to extract a nice 3D model of the action in the video, and then they ran a RL algorithm that got reward based on how closely it could match that model over time. And they were able to train + augment it to work with varying interferences in the environment, plus with different body types. I'd love to see essentially just pushing this further. Maybe see if you can fit it to extremely different body types (spider-type, dog-type). Or just cool stuff, like throwing the trained models into a video game and allow someone to give high-level controls and have the agent follow them in a physically simulated way.

### [QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/pdf/1806.10293.pdf)

Best paper winner at CoRL. The TLDR is they got really good at picking things up with robot arms. It's important work to push the boundaries and build practical systems out of what works. But on the RL side, I think the algorithm they use is of the most interest. It's a Q-learning algorithm, which typically I haven't seen in recent publications in favor of actor-critic methods.

It's actually pretty simple. They are in a sense still doing actor-critic. Except instead of value estimator, they are using a Q-value estimator, and instead of an actor NN, they are just fitting a gaussian using cross-entropy to estimate the action that produces the highest value from the Q-values given a state. It's the same pipeline, except they're removing a lot of the weirdness of optimizing a gradient on top of another changing network. At least in theory, from a thought-expiriment and empirical point of view. This is pretty cool, I think I'll try this out on some atari games and see if it translates nicely.

### [Reinforcement Learning with Prediction-Based Rewards](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/)

This blog post is essentially the combined work of two papers: [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/pdf/1808.04355.pdf) and the work built on it by [Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf). The general crux of it is that if the agent can accurately predict what will happen, it will avoid those scenarios in favor of states that it doesn't know. This is formalized by adding the difference between observed and predicted states as a reward.

I like this a lot, because it's driving towards the idea of learning in general rather than learning for a reward. It'd be cool to see this work continued in the context of curiosity guided by reward (maybe multiply estimated values with error of prediction values for reward?).

## Neural Networks

### [Learning Hierarchical Information Flow with Recurrent Neural Modules](https://arxiv.org/pdf/1706.05744.pdf)
![](https://github.com/ltecot/papers/blob/master/ThalNet_diagram.png)

I think this picture from the paper is essentially all you need to get the gist of the idea. There's definitely more to the implementation of the model, but personally I didn't find that or the results particularly interesting. It's the concept of having a heirarchy or recurrent units that I found interesting, and they were inspired to try this out by the thalamus. I would love to see work around this on seeing if you can observe this type of compartementalization emerge training recurrent networks that don't force the constraint. Perhaps something around neural spiking, or using some form of other penalization than encourages heirarchy.

### [Capacity and Trainability in Recurrent Neural Networks](https://arxiv.org/pdf/1611.09913.pdf)
Overview of a bunch of different RNN architectures. They do a variety of analysis with hyperparameters, number of parameters, number of units, etc. Most of the analysis centers around the network's ability to reproduce binary input it saw in the past, and they use that get the bits per parameter measure of the network. They essentially found that most networks get 3-6 bits per parameter, though strictly speaking this measure might not be particularly useful because the LSTM and GRU networks that had worse memory per parameter train better on more practical tasks. And a little interesting tidbit I liked is they say bioligcal synapses were calulated to have similar capacity, 4.7 bits. Although the this paper is more focused around analysing RNN architectures for choosing and training current models, I'd be interested in seeing similar analysis in RNN's ability to encode concepts, causal relations, etc.

### [Shallow Learning For Deep Networks](https://openreview.net/pdf?id=r1Gsk3R9Fm)
![](https://github.com/ltecot/papers/blob/master/shallow_learning.png)

An iterative process of training neural networks (this this case, a CNN). The picture above from their paper is pretty illustraive of how it works. They essentially train one layer at the time. I didn't read very deeply into their theorems and details, but the general theme seems to be that this allows them to have the nice training properties of single layer neural networks, because they can treat the previous layer just as a regular input. And this allows them to get pretty good results with a smaller network. This also continues my interest on seeing if there can be some form of general network, where neurons and their architecture aren't constrained and are learned / altered through time. And also if they can be trained to represent useful concepts locally, in order to account for a vastly changing goal.

### [Towards deep learning with spiking neurons in energy based models with contrastive Hebbian plasticity](https://arxiv.org/pdf/1612.03214.pdf)

An implementation of networks optimized through hebbian-like local plasticity rules. They essentially formulate the learning problem as loss constrained by an energy equation, and then solve it through a relatively simple update rule. Seems promising and interesting, though to be honest I don't understand some of the math so I'll need to spend a little longer with this. The algorithm steps on page 4 break it down in very simple terms. There's essentially a feed-forward where you subtract from weights that have any correlation, and then a feed-backwards where you fix the output, let the energy minimize for a while and then add any correlations. You do this by essentially continually minimizing the energy equation and alternating the weights that correspond to how important the difference in output is between 0 and some positive value.

This is really interesting but the intuition is a little lost on me. The paper this work is built off of the paper [Towards Biologically Plausible Deep Learning](https://arxiv.org/pdf/1502.04156.pdf), which has a really nice diagram on the third page that illustrates how synaptic learning works in theory. In essence, you want to decrease correlation between a connection that causes another unit to fire, but if there is simple correlation and not temporal causation, you want to decrease those weights. (Or at least that's my general understanding, I didn't read this paper in full yet.) I'd like to see these ideas applied to a variety of other areas. Looking at optimization for this type of learning, maybe simpler rules, applying to RL, etc.

### [Differentiable plasticity: training plastic neural networks with backpropagation](https://eng.uber.com/differentiable-plasticity/)

This blog post and the attached paper basically specify a form of learning that combines the hebbian learning concept with traditional deep learning practices. Basically they train the network with two parameters: typical activation weights, and a term for the weight of the hebbian value, which is essentially a running average of the tendency for both neurons to be activated at the same time (they use a little extra trick to prevent decay of the hebbian value of an inactive neuron over long time horizons). They train the weights with typical SGD methods, and then the hebbian parameter is a more plastic term to allow better expressibility and memory. And it seems to work well in the contexts of the tasks they tried out.

There are some interesting ideas here. Firstly, the idea of combining traditional static weights with backprop with a more plastic learning method. Their anecodtal justification for this is saying that this initial learning is like the equivalent of going through natural selection to get the human brain, and that the hebbian part is the learning over a lifetime. I can't really speak on that, but I think it's a nice trick to help bootstrap new optimization methods. Using hebbian and local-plasticity methods right now isn't as tried and tested as SGD, and I'd guess that a lot of the more highly supervised problems won't really benefit from it. So this setup likely makes it easier to get to a point where you can observe interesting behaviors and performance improvments. You could in theory place any sort of hebbian rule you like into this model, though I'm not sure how stable it will be if you throw something like what was used in the [contrastive Hebbian plasticity](https://arxiv.org/pdf/1612.03214.pdf) paper. In theory it's just a method of constraining the hypothesis space, so as long as you know that your early training is representative of what the model will mostly be dealing with I'm ok with that.

They mention papers like [MAML](https://arxiv.org/abs/1703.03400) and [SNAIL](https://arxiv.org/pdf/1707.03141.pdf) papers, saying that they got similar performance with much fewer parameters. I haven't read either in great detail, but the general gist I get with MAML is that they train over a wide variety of tasks, and with SNAIL they add a bunch of layers with convolutions through time and attention modules to create more expresibility in terms of accessing previous inputs. Although pushing this stuff is important, I do like the author's implied point here. I think many of these problems would benefit from exploring other forms of optimization + networks and understanding which ones create which behaviors. Not just seeing what we can do by throwing more data and more layers + different types of expressability at our models, but investigating what rules allow desired behaviors to be emergent.

Though I will say that I'm not totally sold on their hebbian learning equation. In the context of memory I can certaintly see how it would be useful, because it encodes patterns that have been seen before by increasing the probability of neurons "firing" if they have fired together before in the past. But in the context of learning, it's essentially just a complex term to increase the expressibility of the model. There's a lot in the area this type of learning dealing with hebbian local plasticity updates (change in update direction based on post and pre synaptic firings, dynamic weights within the hebbain learning model, constraints for computational efficiency) that would be nice to explore here. Also, I'm not really a fan of how they don't really give any theoretical justification for their method, and they kinda waive around claims. But that's more of a research-level nitpick from me. I'm certaintly a fan of having both a healthy dose of the theoretical and empirical, so hopefully the community as a whole around these concepts builds on this.

###[Differentiable neural computers](https://deepmind.com/blog/differentiable-neural-computers/)

I only skimmed the blog, (and in ML community standards this paper is a little old) but I wanted to write a quick thing about this because I really like it. I'd expect there's also a lot of new stuff out there around this paper that I should read.

The TLDR here is essentially you make a policy that decides operations to take given a set of operations as actions and a set of memory it has access to to write and read. I like thinking about different rules of operation and optimization for neural networks that provide better memory and modeling of the environment, but I think the idea of giving it access to a memory bank is a cool idea. It sort of implicitly gives the model a lot of memory to work with. And you can fit this model into typical reinforcment learning setups. I'm not sure with what this has been proven to work with yet, but I'd like to expiriment with this more.

## Logic, Relational Learning, and Causal Inference

### [A Semantic Loss Function for Deep Learning Under Weak Supervision](https://web.cs.ucla.edu/~guyvdb/papers/XuLLD17.pdf)

Essentially they constrained a NN to also include loss from breaking logical sentences, presumably that decribe the rules of the task in a logical way (I didn't read too deep). Good really straightforward example of why this type of reasoning is important to try to include in modern methods. They didn't neccisarily see better results, but they were able to achieve optimal solutions faster and with less data.

## Quantum Computing
Currently in a superposition of empty and full.

## 404 Category Not Found

### Resources

#### [OpenAI Request for Research 2.0](https://blog.openai.com/requests-for-research-2/)
Answer the call =)

#### [David Silver's Introduction to Reinforcment Learning](https://youtu.be/2pWv7GOvuf0)
This is what I watched to first learn about RL. Silver is a great teacher, highly recommend to get an overview of the basics of RL.

#### Abbel's [NIPS 2017 Keynote](https://www.facebook.com/nipsfoundation/videos/1554594181298482/) and [updated slides](https://www.dropbox.com/s/uwq7eq8vtmiyr9k/2018_09_09%20--%20Columbia%20--%20Learning%20to%20Learn%20--%20Abbeel.pdf?dl=0)
Abbeel and the other Berkeley folks are have done and continue to do impressive work! Check them out. Good insight into lots of state of the art stuff, even if it's perhaps a little biased.

#### [Michael Nielsen's Quantum Computing Course](https://www.youtube.com/watch?v=X2q1PuI2RFI&list=PL1826E60FD05B44E4&index=1)
Nielsen is mostly known for his very popular online book on neural networks, but he also has excellent material on understanding quantum computing!

#### [Rigetti Intro to Quantum Computing](https://pyquil.readthedocs.io/en/stable/intro.html)
Rigetti is a cool company, and they have solid intros to quantum computing and how to use their tools.

#### [WildML](http://www.wildml.com/author/dennybritz/)
Denny Britz is a pretty cool guy, and I like his posts. Honstly I don't look at it too much, but there was a recent post about using [RL in stock trading](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/) that caught my eye. He brought up a good point about a need for exploration methods as we look at enviroments with an extremely small percent of good moves. Perhaps research in causality will help that out.

#### [Berkeley's Deep Learning Class](http://rail.eecs.berkeley.edu/deeprlcourse/)
I haven't actually followed along on this course, but the material looks good, and it's been taught by Abbeel and Levine. I'd like to work through it soon.

#### [Stanford CS231n](http://cs231n.stanford.edu/index.html)
Haven't done this totally as well, but also seems good. And recommended by a lot of people.

#### [Andrej Karpathy Blog](http://karpathy.github.io/)
In the spirit of continuing to jump on that hype train, Karpathy also has a lot of good writeups. I specifically enjoyed his recurrent neural network one.

### That's Pretty Dope

#### [OpenAI Dexterity](https://blog.openai.com/learning-dexterity/)
LSTM for dayz

#### [OpenAI 5](https://blog.openai.com/openai-five/)
LSTM for **_dayzzz_**

#### [Quantum STAQ Project](https://pratt.duke.edu/about/news/duke-lead-15-million-program-create-first-practical-quantum-computer)
For those of you who are consumed by software, [Fred Chong](https://people.cs.uchicago.edu/~ftchong/) is your guy.

#### [Troubling Trends in Machine Learning Scholarship](https://arxiv.org/abs/1807.03341)
Great read about problems in the ML research community.

### Papers to Read
* https://arxiv.org/pdf/1802.04821.pdf Full paper on envolved policy gradients
* https://arxiv.org/abs/1703.03400 Original meta-learning paper
* https://arxiv.org/pdf/1709.02023.pdf
* https://arxiv.org/pdf/1807.09341.pdf
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3792559/
* https://arxiv.org/pdf/1808.07804.pdf Claims to use some ideas from causal literature, might be good to look at. About transfer learning
* https://arxiv.org/pdf/1707.03141.pdf SNAIL paper. Continuation of work that uses LSTMs to implement policy gradients.
* https://arxiv.org/pdf/1802.01557.pdf One-shot learning from human example. Sounds impressive.
* https://arxiv.org/pdf/1802.06070.pdf Title caught my eye. Sounds like might have insights to Hebbian learning
* https://arxiv.org/pdf/1710.11622.pdf Looks like potentially a cool proof
* https://arxiv.org/abs/1707.01495 Hindsight experience replay. I think I already kinda get how they did this, and I feel like it's only applicable to a specific set of tasks, but still good to read.
* https://arxiv.org/abs/1709.04326 Opponent awareness. This ties into a lot of work about forgetting nash equlibriums and how we can gain value by modeling how we think other people think plus learn.
* https://arxiv.org/abs/1707.00183 Teacher-student. Same as above, idea of modeling agents you interact with.
* https://arxiv.org/pdf/1808.00177.pdf OpenAI dexterity paper
* https://arxiv.org/pdf/1806.07811.pdf Quanquan Gu's paper on a new SGD with reduced variance. NIPS 2018
* https://arxiv.org/pdf/1802.09025.pdf Learning quantum states, AKA minizing number of quantum trails. NIPS 2018
* https://arxiv.org/pdf/1711.02301.pdf Irpan
* https://arxiv.org/pdf/1806.10293.pdf Irpan
* https://arxiv.org/pdf/1709.07857.pdf Irpan
* https://arxiv.org/pdf/1611.01838.pdf Entropy SGD
* https://blog.openai.com/glow/ generative models, looks trippy
* https://blog.openai.com/reptile/ OpenAI's system for training meta models.
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3792559/ Neural spiking stuff
* https://arxiv.org/pdf/1710.02298.pdf Rainbow. I think just an ensemble of a bunch of DQN stuff?
* https://arxiv.org/pdf/1805.11593.pdf RL paper from google to get good performance over all atari games.
* https://arxiv.org/abs/1410.5401 Neural Turing Machine
* https://arxiv.org/abs/1806.09729 Quantum backprop. Gotta get in all those buzzwords.
* http://www1.icsi.berkeley.edu/~shastri/psfiles/nc2000.pdf Hebbian + causal link dump
* https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006227 Hebbian + causal link dump
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4006178/ Hebbian + causal link dump
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4563904/ Hebbian + causal link dump
* https://www.vicarious.com/wp-content/uploads/2018/01/AAAI18-pixelworld.pdf Vicarious AAAI. Seems interesting
* https://arxiv.org/pdf/1712.09913.pdf Cool paper on visualizing loss landscapes
* https://arxiv.org/pdf/1803.10760.pdf Memory paper Mark suggested
* https://arxiv.org/pdf/1603.01121.pdf Imperfect information self play. About modeling minds and nash eq.
* https://www.cell.com/neuron/fulltext/S0896-6273(18)30543-9 neuroscience + RNNs dynamics
* https://www.nature.com/articles/nn.3405 neuroscience + RNNs dynamics
* https://www.biorxiv.org/content/early/2017/09/01/183632 neuroscience + RNNs dynamics
* https://www.mitpressjournals.org/doi/abs/10.1162/NECO_a_00409?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed neuroscience + RNNs dynamics
* https://blog.openai.com/amplifying-ai-training/ Seems to be a form of imitation learning via breaking into smaller known tasks, and building an ensemble of known tasks by making the new one part of ensemble.
