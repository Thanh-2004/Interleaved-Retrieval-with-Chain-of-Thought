import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

class LLM(ABC):
    """Abstract base class for language model implementations."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response based on the provided prompt.
        
        Args:
            prompt: The input text to generate a response for
            **kwargs: Additional parameters for the model
            
        Returns:
            A string containing the generated response
        """
        pass

class OpenRouterLLM(LLM):
    """Implementation of LLM using OpenRouter.ai's API."""
    def __init__(self, model_name: str):
        """Initialize the OpenRouter LLM.
        
        Args:
            model_name: Name of the OpenRouter model to use 
        """
        self.model_name = model_name
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Import here to avoid dependency if not using this class
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            )

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using OpenRouter.ai's API.
        
        Args:
            prompt: The input text to generate a response for
            **kwargs: Additional parameters to pass to the OpenRouter API
            
        Returns:
            The generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if provided
        system_message = kwargs.pop("system_message", None)
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        
        # Set default parameters but allow overrides
        params = {
            # temperature: 0.7,
            # max_tokens: 1000,
        }
        params.update(kwargs)
        
        # Create the completion
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            **params
        )

        return response.choices[0].message.content


class GroqLLM(LLM):
    """Implementation of LLM using Groq's API."""
    
    def __init__(self, model_name: str):
        """Initialize the Groq LLM.
        
        Args:
            model_name: Name of the Groq model to use (e.g., "llama-3.3-70b-versatile")
        """
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Import here to avoid dependency if not using this class
        from groq import Groq
        self.client = Groq(api_key=self.api_key)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using Groq's API.
        
        Args:
            prompt: The input text to generate a response for
            **kwargs: Additional parameters to pass to the Groq API
            
        Returns:
            The generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if provided
        system_message = kwargs.pop("system_message", None)
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        
        # Set default parameters but allow overrides
        params = {
            # temperature: 0.7,
            # max_tokens: 1000,
        }
        params.update(kwargs)
        
        # Create the completion
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            **params
        )
        
        return response.choices[0].message.content


class OpenAILLM(LLM):    
    def __init__(self, model_name: str):
        """Initialize the OpenAI LLM.

        Args:
            model_name: Name of the OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        """
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Import here to avoid dependency if not using this class
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using OpenAI's API.
        
        Args:
            prompt: The input text to generate a response for
            **kwargs: Additional parameters to pass to the OpenAI API
            
        Returns:
            The generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if provided
        system_message = kwargs.pop("system_message", None)
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})
        
        # Set default parameters but allow overrides
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        params.update(kwargs)
        
        # Create the completion
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            **params
        )
        
        return response.choices[0].message.content

# Example usage:
if __name__ == "__main__":
    # Create a Groq LLM instance with the model name
    llm = GroqLLM(model_name="llama-3.3-70b-versatile")
    
    # Generate a response
    response = llm.generate_response(
        prompt="Tell me a joke",
        system_message="You are an AI engineer with good sense of humour",
        temperature= 0.7,
        max_tokens= 1000
    )
    
    print(response)
