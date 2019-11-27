from .evaluation import FakeNewsTextEvaluator

evaluator = FakeNewsTextEvaluator('Quartos limpos e amplos, ótima estrutura e localização, recepcionistas muito educadas. O hotel possui dois restaurantes, academia, internet Wi-Fi (com opção gratuita e paga), piscina e sistema de transfer gratuito para o metrô e o shopping paulista.')
result = evaluator.predict()
print(result)