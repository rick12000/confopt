# Coding Style Guidelines

- Use pytest for all testing, use unittest for mocking.
- Use pytest.mark.parametrize for testing functions with categorical input values:
    - For literals, you should automatically cycle through all possible Literal values in your parametrization.
    - For ordinal categories or discrete inputs (eg. n_observations, n_recommendations, etc.) pick some sensible ranges (eg. 0 if allowed, or minimum otherwise, then a sensible every day value, say 10, then a very large value, if it's not computationally expensive, say 1000).
- Mock external APIs and I/O only, do not use mocking as a crutch to abstract away components who's behaviour you need to test.
- Use fixtures to store toy data, mocked objects or any other object you plan to reference in the main tests, particularly if it will be used more than once. If the toy data is small and specific to the one test it's called in, it's ok to define it inside the test function.
- Never define nested functions (function def is inside another function) unless explicitly required because of scope (eg. nested generator builders).
- Avoid defining helper functions at the top of a test module, tests should be simple and mostly check existing methods' outputs. Very complex tests may require helper functions, but this should be limited.
- ALL fixtures need to be defined in the tests/conftest.py file, NEVER define them directly in a test module.
- Do not test initialization of classes. Do not use asserts that just check if an attribute of a class exists, or is equal to what you just defined it as, these are bloated tests that accomplish little, but add maintenance cost.
- If you're testing a function or method that returns a shaped object, always check the shape (should it be the same as the input's? Should it be different? Should it be a specific size based on the inputs you passed to the function? etc. based on these questions formulate asserts that check those shape aspects)
- Test the intent behind a function or method, not form or attributes. Read through the function or method carefully, understand its goals and approach, then write meaningful tests that check quality of outputs relative to intent.
- Do not add strings after asserts, eg. do NOT do this:
    assert len(final_alphas) == len(initial_alphas), "Alpha count should remain consistent"
    after any assert statement, it should just be assert len(final_alphas) == len(initial_alphas)
- Keep comments to a minimum, comments should just explain more obscure asserts or tests.
- Each unit test should be a function, functions should not be grouped in testing classes and should not have self attributes.
- When testing mathematical functions, understand the derivations and test assumptions and outputs given mathematical constraints and theory.
- Do not write excessive amounts of tests, focus on the most important aspects of each function.
- Avoid lenghty code repetition. If multiple tests share the same set ups or fixture processing but only differ in asserts, join them in a single test and add comments before each assert.
