import tensorgraph as tg

tg.Graph().as_default()

x1 = tg.Variable(1, 'x1')
x2 = tg.Variable(0, 'x2')
session = tg.Session()
print(session.run(tg.NAND(x1, x2)))
print(session.run(tg.AND(x1, x2)))
print(session.run(tg.OR(x1, x2)))
print(session.run(tg.XOR(x1, x2)))

