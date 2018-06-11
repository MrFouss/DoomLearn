import tensorflow as tf
# Il semblerait que ce soit maintenant tf.contrib.layers
import tensorflow.contrib.slim as slim

# Fortement repris de https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
# Avec l'aide de https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

# On décrit ici une classe qui représente un réseau de neurones capable d'apprentissage renforcé sur Doom. On prend en entrée l'écran du jeu, et en sortie une action.
class QNetwork():

    # h_size est la profondeur de la dernière couche de convolution
    def __init__(self, h_size):

        # On prend en entrée une frame du jeu de dimension 21168 (TODO prendre les vraies dimensions). Les valeurs sont sont sous forme séquentielle (1 dimension).
        # TODO essayer de remplacer None par 1 -> batck axis ?
        self.scalarInput =  tf.placeholder(shape=[None,21168], dtype=tf.float32)

        # On remet l'image dans les bonnes dinensions (image de 84 x 84) (TODO utiliser les bonnes dimensions)
        # -1 servirait à déduire la taille dans la dimension concernée à partir des dimensions de l'input et de la taille dans les autres dimensions
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1,84,84,3])

        # Couches de convolution ...
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        # Normalement, on arrive à la dernière couche avec une sortie de dimension [-1, 1, 1, h_size]
        # Avec tf.shape(...), on pourrait connaitre la dimension d'une couche au runtime (pour automatiquement mettre un kernel à la dimension de l'input et bien sortir une image de côté 1).
        self.conv4 = slim.conv2d( \
            inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
        
        # On applati la dernière sortie de la dernière convolution pour arriver en dimension [-1, h_size] (la première dimension (batch) est préservée)
        self.flatConv4 = slim.flatten(self.conv4)

        # Couche complètement connectée avec des poids. Elle est initialisée aléatoirement (TODO voire initialisation de Xavier). TODO utiliser le vrai nombre de sorties (i.e. le nombre d'actions possibles)
        # On peut voir ça comme une matrice qui, à chaque entrée (parmis les h_size), associe une action (ici 4 actions).
        W = tf.Variable(tf.random_uniform([h_size,4],0,0.01))

        # On calcul le gain de faire chaque action
        self.Qout = tf.matmul(self.flatConv4,W)

        # L'action choisie est celle qui a le gain max la dimension 1 (dimension 0 étant le batch). Cette couche retourne l'index de l'emplacement max.
        self.predict = tf.argmax(self.Qout,1)



        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        # self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        # self.streamA = slim.flatten(self.streamAC)
        # self.streamV = slim.flatten(self.streamVC)
        # xavier_init = tf.contrib.layers.xavier_initializer()
        # self.AW = tf.Variable(xavier_init([h_size//2,env.actions]))
        # self.VW = tf.Variable(xavier_init([h_size//2,1]))
        # self.Advantage = tf.matmul(self.streamA,self.AW)
        # self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        # self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        # self.predict = tf.argmax(self.Qout,1)
        
        # TODO réécrire et commenter ce qui suit
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        # self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        # self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        # self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)
        
        # self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        # self.td_error = tf.square(self.targetQ - self.Q)
        # self.loss = tf.reduce_mean(self.td_error)
        # self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # self.updateModel = self.trainer.minimize(self.loss)