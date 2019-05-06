import numpy as np
# from actor_network import ActorNet
# from critic_network import CriticNet
from actor_network_bn import ActorNet_bn
from critic_network_bn import CriticNet_bn
from grad_inverter import grad_inverter
from RM import ReplayMemory

RM_size = 100000
Batch_Size = 32
steps = 200
Gamma = 0.99  # Discount Factor

class DDPG:

    def __init__(self, num_of_states, num_of_actions):
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.critic_net = CriticNet_bn(self.num_of_states, self.num_of_actions)
        self.actor_net = ActorNet_bn(self.num_of_states, self.num_of_actions)

        # Ορισμός μέγιστης και ελάχιστης δράσης και κάλεσμα του grad inverter
        action_max = 35
        action_min = 0
        action_bounds = [action_max, action_min]
        self.grad_inverter = grad_inverter(action_bounds)

    def evaluate_actor(self, state_now):
        return self.actor_net.evaluate_actor(state_now)



    # Συνάρτηση που θα εκπαιδεύει το μοντέλο
    def model_train(self, RM):
        # RM = ReplayMemory(100000)
        # RM.minibatches(self, Batch_Size)
        # Σχηματισμός της επόμενης δράσης π(s',W)
        n_s_b = [RM.minibatches(Batch_Size)[i][1] for i in range(Batch_Size)]
        # print(n_s_b)



        # Υπολογισμός του reward και προσθήκη του στο άδειο array παρακάτω
        self.y_i_batch = []
        self.next_action_batch = self.actor_net.evaluate_target_actor(n_s_b)
        # Σχηματισμός του Q_t ^i (s',a',W)
        # print(self.next_action_batch.shape)
        # print(self.next_action_batch)
        # print(self.next_state_batch.shape)
        # print(self.next_state_batch)
        q_next = self.critic_net.evaluate_target_critic(RM.next_state_batch, self.next_action_batch)
        for i in range(0, Batch_Size):

            if i == Batch_Size:
                self.y_i_batch.append(RM.reward_batch[i])

            else:
                self.y_i_batch.append(RM.reward_batch[i] + Gamma*q_next[i][0])

        self.y_i_batch = np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch, [len(self.y_i_batch), 1])
        # print(" y_i_batch =",  self.y_i_batch)

        # Update του critic (Βήμα 16 του αλγορίθμου):
        self.critic_net.train_critic(RM.current_state_batch, RM.action_batch, self.y_i_batch)

        # Update του actor:

        # Υπολογισμός ενός action από το δίκτυο του actor που θα χρησιμοποιηθεί για το gradient:
        action_for_gradient = self.actor_net.evaluate_actor(RM.current_state_batch)

        # Υπολογισμός του gradient inverter (Βήμα 17 του αλγορίθμου):
        self.dq_da = self.critic_net.compute_delQ_a(RM.current_state_batch, action_for_gradient)
        self.dq_da = self.grad_inverter.inverter(self.dq_da, action_for_gradient)
        # print(self.dq_da)


        # Update του actor (Βήμα 19 του αλγορίθμου):

        self.actor_net.train_actor(RM.current_state_batch, self.dq_da)

        # Τελικό update του target actor & critic
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
        return self.y_i_batch
