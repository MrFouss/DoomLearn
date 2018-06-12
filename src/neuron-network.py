import tensorflow as tf
# Il semblerait que ce soit maintenant tf.contrib.layers
import tensorflow.contrib.layers as slim

# Fortement repris de https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
# Avec l'aide de https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

# On décrit ici une classe qui représente un réseau de neurones capable d'apprentissage renforcé sur Doom. On prend en entrée l'écran du jeu, et en sortie une action.
class QNetwork():

    # h_size est la profondeur de la dernière couche de convolution
    def __init__(self, h_size, randActionProba, y):

        #### Champs de classe

        self.randActionProba = randActionProba
        self.y = y

        #### Mise en place du réseau de neurone

        # On prend en entrée une frame du jeu de dimension 21168 (TODO prendre les vraies dimensions). Les valeurs sont sont sous forme séquentielle (1 dimension).
        # TODO essayer de remplacer None par 1 -> batch axis ?
        self.scalarInput =  tf.placeholder(shape=[None, 21168], dtype=tf.float32)

        # On remet l'image dans les bonnes dimensions (image de 84 x 84) (TODO utiliser les bonnes dimensions)
        # -1 servirait à déduire la taille dans la dimension concernée à partir des dimensions de l'input et de la taille dans les autres dimensions
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])

        # Couches de convolution ...
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3, 3],stride=[1, 1],padding='VALID', biases_initializer=None)
        # Normalement, on arrive à la dernière couche avec une sortie de dimension [-1, 1, 1, h_size]
        # Avec tf.shape(...), on pourrait connaitre la dimension d'une couche au runtime (pour automatiquement mettre un kernel à la dimension de l'input et bien sortir une image de côté 1).
        self.conv4 = slim.conv2d( \
            inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding='VALID', biases_initializer=None)
        
        # On applati la dernière sortie de la dernière convolution pour arriver en dimension [-1, h_size] (la première dimension (batch) est préservée)
        self.flatConv4 = slim.flatten(self.conv4)

        # Couche complètement connectée avec des poids. Elle est initialisée aléatoirement (TODO voire initialisation de Xavier). TODO utiliser le vrai nombre de sorties (i.e. le nombre d'actions possibles)
        # On peut voir ça comme une matrice qui, à chaque entrée (parmis les h_size), associe une action (ici 4 actions).
        self.W = tf.Variable(tf.random_uniform([h_size, 4], 0, 0.01))

        # On calcul le gain de faire chaque action
        self.Qout = tf.matmul(self.flatConv4, self.W)

        # L'action choisie est celle qui a le gain max la dimension 1 (dimension 0 étant le batch). Cette couche retourne l'index de l'emplacement max.
        self.predict = tf.argmax(self.Qout, 1)



        #### Mise en place de la partie d'apprentissage du réseau de neurone

        # TODO remplacer la taille par le bon nombre d'actions
        # On envoi en entrée les valeurs de Qout mises à jour (à partir de la récompense obtenu de l'environnement). En somme, on essaye de tendre vers les estimations de récompense (Qout) les plus justes pour toutes les actions.
        self.nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)

        # On calcul l'erreur au carré entre Qout et nextQ (estimé vs. mesuré)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))

        # C'est le module de méthode d'apprentissage, ici un gradient descend
        # TODO voire aussi AdamOptimizer
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        # On définit l'objectif de l'apprentissage. Ici, on veut réduire l'ecart entre Qout (estimé) nextQ (mesuré).
        # c'est ce module que l'on calcule quand on veut effectuer une propagation arrière dans tout le réseau de neurone
        self.updateModel = self.trainer.minimize(self.loss)


    def train_network(self, env):

        # TODO décommenter avec les fonctions de l'environnement
        pass

        # Calculer l'action choisie (a) et les récompenses évaluées (allQ) en envoyant en entrée une frame de jeu
        # a,allQ = sess.run([self.predict,self.Qout],feed_dict={self.scalarInput:inputFrame)

        # On a une petite probabilité de choisir une action aléatoire afin d'explorer les possibilités d'action
        # if np.random.rand(1) < self.randActionProba:
            # TODO extraire une action aléatoire de l'environnement
            # a[0] = env.action_space.sample()
        
        # Faire l'action choisie et récupérer la récompense
        # r = env.step(a[0])

        # Calculer les récompenses évaluées (Q1) en envoyant en entrée la frame suivante du jeu après avoir effectué l'action choisie
        # Q1 = sess.run(Qout,feed_dict={self.scalarInput:nextInputFrame)

        # On calcule ici les récompenses mises à jour (targetQ) comme suit :
        # targetQ = r + y*maxQ1 (la récompense de l'action + y * la plus grande récompense estimée à la frame suivant)
        # En un sens, on optimise les décisions d'action avec un coup d'avance
        # maxQ1 = np.max(Q1)
        # targetQ = allQ
        # targetQ[0,a[0]] = r + y*maxQ1

        # Enfin, on met à jour le réseau à partir de l'erreur entre récompense mesurée et récompense obtenue
        #_,W1 = sess.run([self.updateModel,self.W],feed_dict={self.scalarInput:inputFrame,self.nextQ:targetQ})
