## __all__ 使用

在Python中，`__all__` 是一个特殊的变量，用于定义模块的公共接口。它是一个字符串列表，包含了模块中所有可以被导入的名称。当使用 `from module import *` 语句时，只有在 `__all__` 列表中的名称会被导入。

例如：

```python
__all__ = ['func1', 'Class1']

def func1():
    pass

def func2():
    pass

class Class1:
    pass

class Class2:
    pass
```

在上面的例子中，如果你使用 `from module import *`，只有 `func1` 和 `Class1` 会被导入，而 `func2` 和 `Class2` 不会被导入。

```python
from module import *

func1()  # This works
Class1()  # This works
func2()  # This will raise a NameError
Class2()  # This will raise a NameError
```

`__all__` 的主要目的是控制模块的命名空间，确保只有指定的名称可以被外部访问。