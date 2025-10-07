import random
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

@dataclass
class Query:
    """Represents a single question-answer template."""
    question: str
    answer_category: str

@dataclass
class Templates:
    """Holds all the template strings for generating task text."""
    definitions: Dict[str, str]
    queries: Dict[str, Query]
    prefix: Optional[str] = None
    capitalize_first_clause: bool = True

@dataclass
class Schema:
    """Represents the blueprint for a binding task using a dataclass for structure."""
    name: str
    items: Dict[str, List[str]]
    templates: Templates
    categories: List[str] = field(init=False)
    max_new_tokens: int = 1
    checker: Callable[[str, str], bool] = lambda neural, causal: causal.lower().strip() in neural.lower().strip()
    matchers: Optional[list] = None

    def __post_init__(self):
        """Derive categories from the items dictionary after initialization."""
        self.categories = list(self.items.keys())

class BindingTask:
    """
    Represents a single instance of a binding task, generated from a schema.

    This class takes a matrix of data (instances and their attributes),
    the categories for those attributes, and templates for generating text.
    It can then create a complete task with a context, question, and answer.
    """
    def __init__(self, data: List[List[str]] | List[Tuple[str, ...]], categories: List[str], templates: Templates):
        if not data:
            raise ValueError("Data matrix cannot be empty.")
        self.data = data
        self.categories = categories
        self.templates = templates
        self.num_instances = len(data)

    def _get_instance_mapping(self, instance_idx: int) -> Dict[str, str]:
        """Creates a dictionary mapping categories to items for a given instance row."""
        return {self.categories[i]: self.data[instance_idx][i] for i in range(len(self.categories))}

    def _format_list(self, items: List[str]) -> str:
        """Formats a list of strings into a natural language list."""
        if len(items) < 2:
            return "".join(items)
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def define_by_key(self, key: str) -> str:
        """Generates a context string using the specified definition key."""
        template = self.templates.definitions[key]
        
        # Check if this is a column-style definition (uses _list format)
        if any(f'{{{cat}_list}}' in template for cat in self.categories):
            # Column-style definition - format with category lists
            items_by_cat = {cat: [row[i] for row in self.data] for i, cat in enumerate(self.categories)}
            all_items_formatted = {cat + '_list': self._format_list(items) for cat, items in items_by_cat.items()}
            return template.format(**all_items_formatted)
        else:
            # Row-style definition - format each instance separately
            clauses = []
            for i in range(self.num_instances):
                mapping = self._get_instance_mapping(i)
                clauses.append(template.format(**mapping))

            if not clauses:
                return ""

            # Conditionally capitalize the first letter of the first clause.
            if self.templates.capitalize_first_clause:
                clauses[0] = clauses[0][0].upper() + clauses[0][1:]
            
            return self._format_list(clauses) + "."

    def format_prefix(self, prefix: str) -> str:
        items_by_cat = {cat: [row[i] for row in self.data] for i, cat in enumerate(self.categories)}
        all_items_formatted = {cat + '_list': self._format_list(items) for cat, items in items_by_cat.items()}
        return prefix.format(**all_items_formatted)

    def generate_task(self, definition_key: str | None = None, query_instance_idx: int | None = None, query_category: str | None = None) -> Dict[str, str]:
        """
        Generates a full task (context, question, answer).
        """
        if definition_key is None:
            definition_key = random.choice(list(self.templates.definitions.keys()))
        
        context = self.define_by_key(definition_key)
        full_context = f"{self.format_prefix(self.templates.prefix)}{context}" if self.templates.prefix else context

        if query_instance_idx is None:
            query_instance_idx = random.randint(0, self.num_instances - 1)

        if query_category:
            selected_query = self.templates.queries.get(query_category)
            if not selected_query:
                raise ValueError(f"No query template found for answer_category: {query_category}")
        else:
            selected_query = random.choice(list(self.templates.queries.values()))

        query_mapping = self._get_instance_mapping(query_instance_idx)
        question = selected_query.question.format(**query_mapping)
        answer = query_mapping[selected_query.answer_category]

        return {
            'context': full_context,
            'question': question,
            'answer': answer
        }
    
class TaskFactory:
    """
    A factory to create BindingTask objects based on predefined schemas.
    """
    def __init__(self, schema: Schema):
        self.schema = schema
        self.categories = schema.categories
        self.templates = schema.templates
        
        if len(self.categories) != 3:
            raise ValueError(f"Schema must contain exactly three categories, but found {len(self.categories)}.")

    def create_task_instance(self, num_instances: int) -> BindingTask:
        """
        Creates a single random data matrix and returns a BindingTask object.
        """
        if any(num_instances > len(self.schema.items[cat]) for cat in self.categories):
            raise ValueError("Cannot create more instances than available unique items.")

        data = []
        sampled_items_by_cat = {
            cat: random.sample(self.schema.items[cat], num_instances)
            for cat in self.categories
        }
        for i in range(num_instances):
            row = [sampled_items_by_cat[cat][i] for cat in self.categories]
            data.append(row)
        
        return BindingTask(data, self.categories, self.templates)

    def create_all_unique_task_instances(self, num_instances: int) -> List[BindingTask]:
        """
        Generates all possible unique task instances for a given number of instances.
        """
        all_tasks = []
        for category in self.categories:
            if len(self.schema.items[category]) < num_instances:
                return []

        item_combos_per_category = [
            list(itertools.combinations(self.schema.items[category], num_instances))
            for category in self.categories
        ]

        for item_sets in itertools.product(*item_combos_per_category):
            fixed_col = item_sets[0]
            permuting_cols = [list(itertools.permutations(s)) for s in item_sets[1:]]
            
            for perms in itertools.product(*permuting_cols):
                data_matrix = list(zip(fixed_col, *perms))
                task = BindingTask(data_matrix, self.categories, self.templates)
                all_tasks.append(task)
                
        return all_tasks
