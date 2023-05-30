
from imitation_model import Imitation
from statistical_model import StatMod


ModelDef = StatMod(lamda=2, mu=1, n=100, requests_number=100, imitation_states=5)

ModelDef.run()

Imitation = Imitation()
Imitation.run(samples=1000, lamda=2, mu=1, n=100, requests_number=100)

