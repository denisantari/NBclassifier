
# Preprocessor
This class implement preprocessor for Indonesian language, including stopword remover and punctuation remover but not limited to stemming Indonesian language.

#### How to use it ?
```
from core.preprocessor import Preprocessor

contents = ['Gue membangun Indonesia',
			'Jogja mulai macet menyerupai Jakarta']
contents = Preprocessor().run(contents)
```

For implement other supervised algorithm, you must extend SupervisedAlgorithm class