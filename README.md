# How to use local LLM

This project is to make it easy to use a local LLM as a backend for a small project
You shouldn´t use this with any other pytorch code as we´re using torch.cuda.empty_cache().
Other than that it seems to work just fine.

This project is intended for local LLMs, but you may want to use an api, so I added support for OpenAI´s api.

The idea behind this project is to use the LLM_connection object, load from a path the model and call its Get_answer function
Works on both Linux and Windows.
