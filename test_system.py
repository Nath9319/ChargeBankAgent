"""
Test System for POML Prompts
This file tests the core functionality without requiring external API calls
"""

import sys
import traceback
from poml_prompts import (
    POMLPromptEngine, 
    POMLPrompt, 
    PromptVariable, 
    PromptFunction,
    CommonPOMLPrompts,
    create_poml_from_template,
    export_poml_to_yaml,
    export_poml_to_json
)


def test_basic_poml():
    """Test basic POML functionality"""
    print("Testing basic POML functionality...")
    
    try:
        # Create a simple prompt
        prompt = POMLPrompt(
            name="test_prompt",
            description="Test prompt for validation",
            system_message="You are a test assistant.",
            user_message_template="Hello {{name}}, how are you?",
            variables=[
                PromptVariable("name", "User's name", "string", True)
            ],
            output_format={
                "type": "object",
                "properties": {
                    "greeting": {"type": "string"}
                }
            }
        )
        
        print(f"✅ Created prompt: {prompt.name}")
        return True
        
    except Exception as e:
        print(f"❌ Basic POML test failed: {e}")
        traceback.print_exc()
        return False


def test_poml_engine():
    """Test POML engine functionality"""
    print("\nTesting POML engine...")
    
    try:
        engine = POMLPromptEngine()
        
        # Register a prompt
        test_prompt = POMLPrompt(
            name="engine_test",
            description="Test prompt for engine",
            system_message="You are a test engine.",
            user_message_template="Test message: {{message}}",
            variables=[
                PromptVariable("message", "Test message", "string", True)
            ],
            output_format={
                "type": "object",
                "properties": {
                    "response": {"type": "string"}
                }
            }
        )
        
        engine.register_prompt(test_prompt)
        print("✅ Registered prompt in engine")
        
        # Render prompt
        variables = {"message": "Hello World"}
        rendered = engine.render_prompt("engine_test", variables)
        
        if "Hello World" in rendered["user_message"]:
            print("✅ Prompt rendering successful")
        else:
            print("❌ Prompt rendering failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ POML engine test failed: {e}")
        traceback.print_exc()
        return False


def test_common_prompts():
    """Test common prompt templates"""
    print("\nTesting common prompt templates...")
    
    try:
        # Test analysis prompt
        analysis_prompt = CommonPOMLPrompts.create_analysis_prompt()
        print(f"✅ Created analysis prompt: {analysis_prompt.name}")
        
        # Test code review prompt
        code_review_prompt = CommonPOMLPrompts.create_code_review_prompt()
        print(f"✅ Created code review prompt: {code_review_prompt.name}")
        
        # Test planning prompt
        planning_prompt = CommonPOMLPrompts.create_planning_prompt()
        print(f"✅ Created planning prompt: {planning_prompt.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Common prompts test failed: {e}")
        traceback.print_exc()
        return False


def test_template_functions():
    """Test template creation functions"""
    print("\nTesting template functions...")
    
    try:
        # Test template creation
        analysis_prompt = create_poml_from_template("analysis")
        print(f"✅ Created analysis prompt from template: {analysis_prompt.name}")
        
        # Test export functions
        yaml_export = export_poml_to_yaml(analysis_prompt)
        if "data_analysis" in yaml_export:
            print("✅ YAML export successful")
        else:
            print("❌ YAML export failed")
            return False
        
        json_export = export_poml_to_json(analysis_prompt)
        if "data_analysis" in json_export:
            print("✅ JSON export successful")
        else:
            print("❌ JSON export failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Template functions test failed: {e}")
        traceback.print_exc()
        return False


def test_prompt_validation():
    """Test prompt validation functionality"""
    print("\nTesting prompt validation...")
    
    try:
        engine = POMLPromptEngine()
        
        # Create prompt with required variables
        validation_prompt = POMLPrompt(
            name="validation_test",
            description="Test validation",
            system_message="You are a validator.",
            user_message_template="Validate: {{required_var}}",
            variables=[
                PromptVariable("required_var", "Required variable", "string", True)
            ],
            output_format={
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            }
        )
        
        engine.register_prompt(validation_prompt)
        
        # Test with missing required variable
        try:
            engine.render_prompt("validation_test", {})
            print("❌ Validation failed - should have required variable")
            return False
        except ValueError:
            print("✅ Validation correctly caught missing required variable")
        
        # Test with valid variables
        try:
            rendered = engine.render_prompt("validation_test", {"required_var": "test"})
            if "test" in rendered["user_message"]:
                print("✅ Validation successful with valid variables")
            else:
                print("❌ Validation failed with valid variables")
                return False
        except Exception as e:
            print(f"❌ Validation failed unexpectedly: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt validation test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 Testing POML Prompt System\n")
    
    tests = [
        test_basic_poml,
        test_poml_engine,
        test_common_prompts,
        test_template_functions,
        test_prompt_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The POML system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)