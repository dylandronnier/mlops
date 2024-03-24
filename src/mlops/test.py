class A:
    def __init__(self, inherited_arg=None):
        self.inherited_arg = inherited_arg

    def method_a(self):
        print("Method A")


def float_to_list(cls):
    class NewClass(cls, A):  # Héritage multiple: d'abord cls, ensuite A
        def __init__(self, *args, new_arg=None, **kwargs):
            inherited_arg = kwargs.pop(
                "inherited_arg", None
            )  # Retirer l'argument spécifique à A
            # Convertir les arguments float en listes
            args = [arg if isinstance(arg, list) else [arg] for arg in args]
            super().__init__(*args, **kwargs)
            self.new_arg = new_arg  # Nouvel argument
            if inherited_arg is not None:
                self.inherited_arg = inherited_arg  # Initialiser l'argument hérité

    return NewClass


# Exemple d'utilisation du décorateur
@float_to_list
class MyClass:
    def __init__(self, values, name):
        self.values = values
        self.name = name


# Test
obj = MyClass(
    values=3.5, name="example", new_arg="new_value", inherited_arg="inherited_value"
)
print(obj.values)  # Output: [3.5]
print(obj.name)  # Output: example
print(obj.new_arg)  # Output: new_value
print(obj.inherited_arg)  # Output: inherited_value
obj.method_a()  # Output: Method A
print(vars(obj))
