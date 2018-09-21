# Papers
Notes and summaries of papers for myself. Not meant to be extremely accurate or thoughtful, so take it with a grain of salt. And along those lines, if you see something that is wrong, please open an issue or pull request!

## Reinforcment Learning

### [Model-Based Reinforcement Learning via Meta-Policy Optimization](https://arxiv.org/abs/1809.05214)
This paper continues the push for what Abbeel calls meta-learning. Specifically it builds off [
Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) by Finn, Abbeel, and Levine. In that paper, they suggest instead of training a RL model on just one task, to update parameters according to rewards from a wide variety of tasks, and then fine-tune the weights during "test" time for each task initialzied from the global optimal. It's a smart way of pushing the power of NN's another step further.

They've pushed this idea further before with [evolved policy gradients](https://blog.openai.com/evolved-policy-gradients/), which prodbably deserves a section of it's own here. The idea being that loss functions are more general and share more similarities between differing tasks, so it'd be best to instead optimize over all environments on a new loss function rather than directly on the policy. The blog post explains this well, but the policy and loss model essentially a form of cooperative network. The policy uses the reward function to try and learn an enviroment as it normally would, it returns the reward it got, and then the reward function uses that value to do gradient descent on it's own parameters.

And finally, this paper goes in another direction of meta-learning. 

### [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://blog.openai.com/evolution-strategies/)
Essentially just trying out a bunch of slightly different policies, choose the best one, rinse and repeat. And this can be faster just because when you're working with a hidden loss like reward, sometimes it's just faster to try it out than computing the gradient. I can see this being a problem with enviroments that involve a lot of other agents or it takes a very long time to play out. Though I only looked at the blog, not the paper and code, so they may have addressed concerns like these more directly there.

### [Meta Learning Shared Hierarchies](https://blog.openai.com/learning-a-hierarchy/)
Essentially just throw RL algorithms on top of eachother. The highest one runs at a longer timestep. It chooses which lower level algorithm to use, and then it's "reward" is just the sum of whatever the lower level model achieved in that time. Pretty cool, at least in the examples they gave seemed to learn good seperations of tasks. I'd like to see someone find counterexamples, or to generaize the system of layers. Essentially "how deep do you go?".

## Recurrent Neural Networks
If you smash the refresh button maybe the state will change.

I will say though that I'm particularly interested in parallels between recurrent networks and how our brain works. Specifically exploring more around Hebbian learning, neural spiking, plasticity in spike timing, long term potentiation / depression, and sleep regularization. But there's probably a lot of literature around all these ideas I haven't seen yet, so I need to read up a lot.

## Quantum Computing
Currently in a superposition of empty and full.

## 404 Category Not Found

### Resources

#### [OpenAI Request for Research 2.0](https://blog.openai.com/requests-for-research-2/)
Answer the call =)

#### Abbel's [NIPS 2017 Keynote](https://www.facebook.com/nipsfoundation/videos/1554594181298482/) and [updated slides](https://www.dropbox.com/s/uwq7eq8vtmiyr9k/2018_09_09%20--%20Columbia%20--%20Learning%20to%20Learn%20--%20Abbeel.pdf?dl=0)
Abbeel and the other Berkeley folks are have done and continue to do impressive work! Check them out. Good insight into lots of state of the art stuff, even if it's perhaps a little biased.

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
LSTM for *dayzzz*

### Papers to Read
* https://arxiv.org/pdf/1802.04821.pdf Full paper on envolved policy gradients
* https://arxiv.org/abs/1703.03400 Original meta-learning paper
* https://arxiv.org/pdf/1709.02023.pdf
* https://arxiv.org/pdf/1807.09341.pdf
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3792559/
* https://arxiv.org/pdf/1808.07804.pdf Claims to use some ideas from causal literature, might be good to look at. About transfer learning
* https://openreview.net/pdf?id=B1DmUzWAW SNAIL paper. Continuation of work that uses LSTMs to implement policy gradients.
* https://arxiv.org/pdf/1802.01557.pdf One-shot learning from human example. Sounds impressive.
* https://arxiv.org/pdf/1802.06070.pdf Title caught my eye. Sounds like might have insights to Hebbian learning
* https://arxiv.org/pdf/1710.11622.pdf Looks like potentially a cool proof
* https://arxiv.org/abs/1707.01495 Hindsight experience replay. I think I already kinda get how they did this, and I feel like it's only applicable to a specific set of tasks, but still good to read.
* https://arxiv.org/abs/1709.04326 Opponent awareness. This ties into a lot of work about forgetting nash equlibriums and how we can gain value by modeling how we think other people think plus learn.
* https://arxiv.org/abs/1707.00183 Teacher-student. Same as above, idea of modeling agents you interact with.
* https://arxiv.org/pdf/1808.00177.pdf OpenAI dexterity paper
