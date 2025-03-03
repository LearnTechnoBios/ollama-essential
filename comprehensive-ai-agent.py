# main.py
import os
import logging
import time
import uuid
import json
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict, field
import unittest
from unittest.mock import Mock, patch
import asyncio
from contextlib import contextmanager
import cProfile
import pstats
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("agent_system")

# ----- CORE COMPONENTS -----

@dataclass
class TaskResult:
    content: str
    success: bool
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourceMetrics:
    cpu_percent: float
    memory_percent: float
    timestamp: float

class ResourceMonitor:
    def __init__(self, logging_interval: int = 60):
        self.metrics: List[ResourceMetrics] = []
        self.logging_interval = logging_interval
        self.last_logged = 0
        
    def capture_metrics(self) -> ResourceMetrics:
        metrics = ResourceMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            timestamp=time.time()
        )
        self.metrics.append(metrics)
        
        # Periodically log metrics
        if time.time() - self.last_logged > self.logging_interval:
            self.log_metrics()
            self.last_logged = time.time()
            
        return metrics
    
    def log_metrics(self):
        if not self.metrics:
            return
            
        latest = self.metrics[-1]
        logger.info(f"Current resources - CPU: {latest.cpu_percent}%, Memory: {latest.memory_percent}%")
    
    def get_average_metrics(self, last_n: int = 10) -> Dict[str, float]:
        if not self.metrics:
            return {"cpu_percent": 0, "memory_percent": 0}
            
        metrics_to_avg = self.metrics[-min(last_n, len(self.metrics)):]
        return {
            "cpu_percent": sum(m.cpu_percent for m in metrics_to_avg) / len(metrics_to_avg),
            "memory_percent": sum(m.memory_percent for m in metrics_to_avg) / len(metrics_to_avg)
        }

@dataclass
class Task:
    description: str
    expected_output: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    result: Optional[TaskResult] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "expected_output": self.expected_output,
            "context": self.context,
            "result": asdict(self.result) if self.result else None
        }

class Agent:
    def __init__(
        self, 
        role: str, 
        goal: str, 
        backstory: str,
        llm: Any = None,
        debug_mode: bool = False
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm or MockLLM()  # Use a mock if no LLM provided
        self.execution_history = []
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(f"agent.{role}")
        self.state: Dict[str, Any] = {}
        self.profiler = cProfile.Profile()
        
    def execute(self, task: Task) -> TaskResult:
        """Execute a task and return the result"""
        self.logger.info(f"Executing task: {task.description}")
        
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Log the beginning of execution
        self._log_execution_event(trace_id, "start", task)
        
        try:
            # Start profiling if in debug mode
            if self.debug_mode:
                self.profiler.enable()
                
            # Generate response from LLM
            prompt = self._create_prompt(task)
            result = self.llm.predict(prompt)
            
            # Create task result
            task_result = TaskResult(
                content=result,
                success=True,
                execution_time=time.time() - start_time,
                metadata={
                    "trace_id": trace_id,
                    "agent_role": self.role,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Store result in task
            task.result = task_result
            
            # Record in execution history
            self._record_execution(task, task_result)
            
            return task_result
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            # Create error result
            error_result = TaskResult(
                content=f"Error: {str(e)}",
                success=False,
                execution_time=time.time() - start_time,
                metadata={
                    "trace_id": trace_id,
                    "agent_role": self.role,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Store result in task
            task.result = error_result
            
            # Record in execution history
            self._record_execution(task, error_result)
            
            return error_result
        finally:
            # Stop profiling if in debug mode
            if self.debug_mode:
                self.profiler.disable()
                
            # Log the end of execution
            self._log_execution_event(trace_id, "end", task)
    
    def _create_prompt(self, task: Task) -> str:
        """Create a prompt for the LLM based on the task"""
        return f"""
        You are a {self.role} with the goal: {self.goal}
        Your backstory: {self.backstory}
        
        TASK: {task.description}
        
        Additional context: {json.dumps(task.context)}
        
        Provide your response:
        """
    
    def _record_execution(self, task: Task, result: TaskResult):
        """Record the execution in history"""
        execution_record = {
            "task_id": task.id,
            "task_description": task.description,
            "result": result.content,
            "success": result.success,
            "execution_time": result.execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_history.append(execution_record)
        
        # Update state with latest execution
        self.state["last_execution"] = execution_record
    
    def _log_execution_event(self, trace_id: str, event: str, task: Task):
        """Log an execution event"""
        self.logger.debug(
            f"Execution {event} - Trace: {trace_id}, "
            f"Agent: {self.role}, Task: {task.id}"
        )
    
    def get_performance_stats(self) -> Optional[pstats.Stats]:
        """Get performance statistics if in debug mode"""
        if self.debug_mode:
            return pstats.Stats(self.profiler)
        return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of all executions"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "average_execution_time": 0
            }
            
        total = len(self.execution_history)
        successful = sum(1 for record in self.execution_history if record["success"])
        avg_time = sum(record["execution_time"] for record in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "average_execution_time": avg_time
        }

class Crew:
    def __init__(
        self, 
        agents: List[Agent], 
        tasks: List[Task],
        max_memory_percent: float = 85.0,
        task_completion_strategy: str = "sequential"
    ):
        self.agents = agents
        self.tasks = tasks
        self.max_memory_percent = max_memory_percent
        self.task_completion_strategy = task_completion_strategy
        self.resource_monitor = ResourceMonitor()
        self.logger = logging.getLogger("crew")
        self.results: Dict[str, TaskResult] = {}
    
    def execute_tasks(self) -> Dict[str, TaskResult]:
        """Execute all tasks according to the specified strategy"""
        self.logger.info(f"Executing {len(self.tasks)} tasks with {len(self.agents)} agents")
        
        start_time = time.time()
        
        try:
            if self.task_completion_strategy == "sequential":
                self._execute_sequential()
            elif self.task_completion_strategy == "parallel":
                asyncio.run(self._execute_parallel())
            else:
                raise ValueError(f"Unknown task completion strategy: {self.task_completion_strategy}")
                
            # Log completion summary
            self.logger.info(
                f"Completed {len(self.results)}/{len(self.tasks)} tasks in "
                f"{time.time() - start_time:.2f} seconds"
            )
            
            return self.results
        except Exception as e:
            self.logger.error(f"Error executing tasks: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def _execute_sequential(self):
        """Execute tasks in sequential order"""
        for i, task in enumerate(self.tasks):
            # Check resources before executing
            self._check_resources()
            
            # Get the appropriate agent for this task
            agent = self._get_agent_for_task(i)
            
            # Execute the task
            result = agent.execute(task)
            
            # Store the result
            self.results[task.id] = result
            
            # Update context for subsequent tasks
            self._update_task_context(task, result)
    
    async def _execute_parallel(self):
        """Execute tasks in parallel"""
        async def execute_task(agent, task):
            # We need to run agent.execute in a separate thread
            # since it's a blocking operation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, agent.execute, task
            )
            return task, result
        
        # Create list of tasks to execute
        tasks = []
        for i, task in enumerate(self.tasks):
            agent = self._get_agent_for_task(i)
            tasks.append(execute_task(agent, task))
        
        # Execute tasks concurrently and process results as they complete
        for completed_task in asyncio.as_completed(tasks):
            task, result = await completed_task
            self.results[task.id] = result
            self._update_task_context(task, result)
    
    def _get_agent_for_task(self, task_index: int) -> Agent:
        """Get the appropriate agent for a task"""
        # Simple round-robin assignment - can be made more sophisticated
        return self.agents[task_index % len(self.agents)]
    
    def _check_resources(self):
        """Check system resources and pause if necessary"""
        metrics = self.resource_monitor.capture_metrics()
        
        if metrics.memory_percent > self.max_memory_percent:
            self.logger.warning(
                f"Memory usage ({metrics.memory_percent}%) exceeds threshold "
                f"({self.max_memory_percent}%). Pausing execution."
            )
            time.sleep(5)  # Wait for memory to be released
    
    def _update_task_context(self, task: Task, result: TaskResult):
        """Update context for subsequent tasks based on result"""
        for next_task in self.tasks:
            if next_task.id == task.id:
                continue
                
            # Add this task's result to context of other tasks
            next_task.context[f"task_{task.id}_result"] = result.content
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get an execution summary for all tasks"""
        successful = sum(1 for result in self.results.values() if result.success)
        
        if not self.results:
            return {
                "total_tasks": len(self.tasks),
                "completed_tasks": 0,
                "successful_tasks": 0
            }
            
        avg_time = sum(result.execution_time for result in self.results.values()) / len(self.results)
        
        return {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.results),
            "successful_tasks": successful,
            "average_execution_time": avg_time,
            "resource_usage": self.resource_monitor.get_average_metrics()
        }

# ----- MOCK LLM FOR TESTING -----

class MockLLM:
    """A mock LLM for testing"""
    def __init__(self, response: str = None):
        self.response = response or "This is a mock response from the LLM."
    
    def predict(self, prompt: str) -> str:
        """Return a mock response"""
        # For testing, we might want to make the response dependent on the prompt
        if "error" in prompt.lower():
            raise Exception("Simulated LLM error")
        return self.response

# ----- APPLICATION-SPECIFIC IMPLEMENTATION -----

class ContentAnalysisSystem:
    """A system for analyzing content sentiment and topics"""
    
    def __init__(self, debug_mode: bool = False):
        # Create agents with different specialties
        self.sentiment_agent = Agent(
            role="Sentiment Analyst",
            goal="Analyze content sentiment accurately",
            backstory="You are an expert at detecting emotional tones in text.",
            debug_mode=debug_mode
        )
        
        self.topic_agent = Agent(
            role="Topic Identifier",
            goal="Identify main topics and themes in content",
            backstory="You are skilled at categorizing content by subject matter.",
            debug_mode=debug_mode
        )
        
        self.summary_agent = Agent(
            role="Content Summarizer",
            goal="Create concise, accurate summaries",
            backstory="You excel at distilling complex content into clear summaries.",
            debug_mode=debug_mode
        )
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for sentiment, topics, and create a summary"""
        
        # Create tasks for the analysis
        sentiment_task = Task(
            description="Analyze the sentiment of the following content. Classify as positive, negative, or neutral, and provide a confidence score between 0 and 1.",
            context={"content": content}
        )
        
        topic_task = Task(
            description="Identify the main topics and themes in the following content. List up to 5 key topics.",
            context={"content": content}
        )
        
        summary_task = Task(
            description="Create a concise summary of the following content in 2-3 sentences.",
            context={"content": content}
        )
        
        # Create a crew to execute the tasks
        crew = Crew(
            agents=[self.sentiment_agent, self.topic_agent, self.summary_agent],
            tasks=[sentiment_task, topic_task, summary_task],
            task_completion_strategy="parallel"  # Execute all tasks in parallel
        )
        
        # Execute the tasks
        results = crew.execute_tasks()
        
        # Compile and return the analysis
        return {
            "sentiment_analysis": results[sentiment_task.id].content,
            "topic_analysis": results[topic_task.id].content,
            "summary": results[summary_task.id].content,
            "execution_metrics": crew.get_execution_summary()
        }

# ----- TESTS -----

class TestAgent(unittest.TestCase):
    """Tests for the Agent class"""
    
    def setUp(self):
        # Create a test agent with a mock LLM
        self.agent = Agent(
            role="Test Agent",
            goal="Testing",
            backstory="A test agent for unit tests",
            llm=MockLLM("Test response"),
            debug_mode=True
        )
    
    def test_agent_execution(self):
        """Test basic agent execution"""
        task = Task(
            description="Test task",
            expected_output="Test response"
        )
        
        result = self.agent.execute(task)
        
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Test response")
        self.assertGreater(result.execution_time, 0)
    
    def test_agent_execution_error(self):
        """Test agent execution with an error"""
        task = Task(
            description="Error task",
            context={"trigger_error": True}
        )
        
        # Use a different mock that raises an exception
        self.agent.llm = MockLLM()
        
        # Patch the _create_prompt method to include "error" to trigger the exception
        with patch.object(self.agent, '_create_prompt', return_value="error in prompt"):
            result = self.agent.execute(task)
            
            self.assertFalse(result.success)
            self.assertTrue("Error:" in result.content)
    
    def test_execution_history(self):
        """Test that execution history is maintained"""
        task = Task(description="Task 1")
        self.agent.execute(task)
        
        task2 = Task(description="Task 2")
        self.agent.execute(task2)
        
        self.assertEqual(len(self.agent.execution_history), 2)
        
        summary = self.agent.get_execution_summary()
        self.assertEqual(summary["total_executions"], 2)
        self.assertEqual(summary["successful_executions"], 2)

class TestCrew(unittest.TestCase):
    """Tests for the Crew class"""
    
    def setUp(self):
        # Create test agents
        self.agent1 = Agent(
            role="Agent 1",
            goal="Testing",
            backstory="Test agent 1",
            llm=MockLLM("Response from Agent 1")
        )
        
        self.agent2 = Agent(
            role="Agent 2",
            goal="Testing",
            backstory="Test agent 2",
            llm=MockLLM("Response from Agent 2")
        )
        
        # Create test tasks
        self.task1 = Task(
            description="Task 1",
            expected_output="Test output 1"
        )
        
        self.task2 = Task(
            description="Task 2",
            expected_output="Test output 2"
        )
    
    def test_sequential_execution(self):
        """Test sequential task execution"""
        crew = Crew(
            agents=[self.agent1, self.agent2],
            tasks=[self.task1, self.task2],
            task_completion_strategy="sequential"
        )
        
        results = crew.execute_tasks()
        
        self.assertEqual(len(results), 2)
        self.assertTrue(self.task1.id in results)
        self.assertTrue(self.task2.id in results)
        
        # Check that context was updated
        self.assertTrue(f"task_{self.task1.id}_result" in self.task2.context)
    
    @patch('asyncio.run')
    def test_parallel_execution(self, mock_run):
        """Test parallel task execution"""
        crew = Crew(
            agents=[self.agent1, self.agent2],
            tasks=[self.task1, self.task2],
            task_completion_strategy="parallel"
        )
        
        # Mock the parallel execution
        crew.execute_tasks()
        
        # Verify that asyncio.run was called
        mock_run.assert_called_once()
    
    def test_resource_monitoring(self):
        """Test that resources are monitored during execution"""
        crew = Crew(
            agents=[self.agent1],
            tasks=[self.task1],
            max_memory_percent=99.0  # Set high to avoid pausing
        )
        
        # Patch _check_resources to verify it's called
        with patch.object(crew, '_check_resources') as mock_check:
            crew.execute_tasks()
            mock_check.assert_called()

class TestContentAnalysisSystem(unittest.TestCase):
    """Tests for the ContentAnalysisSystem"""
    
    def setUp(self):
        self.analysis_system = ContentAnalysisSystem(debug_mode=True)
        
        # Configure mock responses
        self.analysis_system.sentiment_agent.llm = MockLLM("Positive sentiment, confidence: 0.85")
        self.analysis_system.topic_agent.llm = MockLLM("Topics: Technology, AI, Python")
        self.analysis_system.summary_agent.llm = MockLLM("This is a summary of the content.")
    
    def test_content_analysis(self):
        """Test content analysis with the system"""
        test_content = "This is a test article about AI and Python programming."
        
        results = self.analysis_system.analyze_content(test_content)
        
        # Check that all analyses were performed
        self.assertIn("sentiment_analysis", results)
        self.assertIn("topic_analysis", results)
        self.assertIn("summary", results)
        
        # Check that execution metrics are included
        self.assertIn("execution_metrics", results)

# ----- USAGE EXAMPLE -----

def main():
    """Main function to demonstrate usage"""
    # Create the content analysis system
    analysis_system = ContentAnalysisSystem(debug_mode=True)
    
    # Sample content to analyze
    content = """
    The new AI chatbot released by OpenTech last week has received overwhelmingly positive reviews 
    from both critics and users. The system, which uses a combination of transformer models and 
    reinforcement learning, has been praised for its natural language understanding and helpful responses. 
    Many users reported that it helped them solve complex programming problems and learn new concepts.
    """
    
    # Analyze the content
    print("Analyzing content...")
    results = analysis_system.analyze_content(content)
    
    # Print the results
    print("\n===== ANALYSIS RESULTS =====")
    print(f"\nSentiment Analysis: {results['sentiment_analysis']}")
    print(f"\nTopic Analysis: {results['topic_analysis']}")
    print(f"\nSummary: {results['summary']}")
    
    # Print execution metrics
    print("\n===== EXECUTION METRICS =====")
    metrics = results['execution_metrics']
    print(f"Total tasks: {metrics['total_tasks']}")
    print(f"Completed tasks: {metrics['completed_tasks']}")
    print(f"Successful tasks: {metrics['successful_tasks']}")
    print(f"Average execution time: {metrics['average_execution_time']:.2f} seconds")
    
    # Get agent performance stats
    print("\n===== AGENT PERFORMANCE =====")
    for agent in [analysis_system.sentiment_agent, analysis_system.topic_agent, analysis_system.summary_agent]:
        summary = agent.get_execution_summary()
        print(f"\n{agent.role}:")
        print(f"  Total executions: {summary['total_executions']}")
        print(f"  Successful executions: {summary['successful_executions']}")
        print(f"  Average execution time: {summary['average_execution_time']:.4f} seconds")

    # Run unit tests
    print("\n===== RUNNING UNIT TESTS =====")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    main()
