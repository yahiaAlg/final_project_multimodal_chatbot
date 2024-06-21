```python
class BubbleSort:
   """A class to implement the bubble sort algorithm."""

   def __init__(self, data):
       """Initialize the object with a list of elements.

       Args:
           data (list): A list of comparable elements.

       Raises:
           TypeError: If 'data' is not a list or contains non-comparable elements.
       """
       if not isinstance(data, list):
           raise TypeError("Input must be a list.")

       for element in data:
           if not self._is_comparable(element):
               raise ValueError("All elements must be comparable.")

       self.data = data

   def _is_comparable(self, element):
       """Check if an element is comparable (i.e., can be compared using < or >).

       Args:
           element: The element to check for comparability.

       Returns:
           bool: True if the element is comparable, False otherwise.
       """
       return hasattr(element, "__lt__") and callable(getattr(element, "__lt__", None))

   def sort(self):
       """Sorts the list using bubble sort algorithm."""
       n = len(self.data)

       for i in range(n):
           try:
               # Perform a single pass of bubble sort
               for j in range(0, n-i-1):
                   if self.data[j] > self.data[j+1]:
                       # Swap the elements
                       self.data[j], self.data[j+1] = self.data[j+1], self.data[j]
           except IndexError:
               raise ValueError("List index out of range.")

   def __str__(self):
       """Returns a string representation of the sorted list."""
       return str(self.data)

# Example usage
if __name__ == "__main__":
   data = [64, 34, 25, 12, 22, 11, 90]
   bubble_sort = BubbleSort(data)

   print("Original list:", data)
   bubble_sort.sort()
   print("Sorted list:", bubble_sort)
```
