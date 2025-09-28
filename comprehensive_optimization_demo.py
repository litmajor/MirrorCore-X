
"""
Comprehensive Optimization Demo for MirrorCore-X
Demonstrates optimization of ALL system parameters
"""

import asyncio
import logging
from mirrorcore_x import create_mirrorcore_system
from mirror_optimizer import ComprehensiveMirrorOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_comprehensive_optimization():
    """Demonstrate comprehensive optimization of all MirrorCore-X parameters."""
    
    print("="*80)
    print("🚀 MirrorCore-X Comprehensive Optimization Demo")
    print("="*80)
    
    try:
        # Create the MirrorCore-X system
        print("📦 Creating MirrorCore-X system...")
        sync_bus, components = await create_mirrorcore_system(dry_run=True)
        
        # Get the comprehensive optimizer
        optimizer = components['comprehensive_optimizer']
        
        print(f"✅ System created with {len(components)} components")
        print(f"🔧 Optimizer covers {len(optimizer.parameter_bounds)} component categories")
        
        # Show all optimizable parameters
        print("\n📊 OPTIMIZABLE PARAMETERS COVERAGE:")
        print("-" * 50)
        
        total_params = 0
        for category, bounds in optimizer.parameter_bounds.items():
            param_count = len(bounds)
            total_params += param_count
            print(f"  {category:<20} : {param_count:>3} parameters")
            if param_count <= 5:  # Show details for smaller categories
                for param in bounds.keys():
                    print(f"    - {param}")
        
        print("-" * 50)
        print(f"  {'TOTAL':<20} : {total_params:>3} parameters")
        
        # Find all optimizable components in the system
        optimizable_components = optimizer.get_all_optimizable_components()
        print(f"\n🎯 Found {len(optimizable_components)} optimizable components:")
        for name in optimizable_components.keys():
            print(f"  - {name}")
        
        # Demonstrate optimization by category
        print("\n🔄 DEMONSTRATING CATEGORY-BASED OPTIMIZATION:")
        print("-" * 50)
        
        categories_to_demo = ['trading', 'analysis', 'risk']
        
        for category in categories_to_demo:
            print(f"\n🎯 Optimizing '{category}' category...")
            results = optimizer.optimize_by_category(category, iterations=5)  # Quick demo
            
            successful = len([r for r in results.values() if r])
            print(f"  ✅ {successful}/{len(results)} components optimized successfully")
            
            for component_name, result in results.items():
                if result:
                    print(f"    {component_name}: {len(result)} parameters optimized")
        
        # Demonstrate full system optimization (abbreviated)
        print(f"\n🚀 COMPREHENSIVE SYSTEM OPTIMIZATION:")
        print("-" * 50)
        print("🔄 Running abbreviated optimization (5 iterations per component)...")
        
        all_results = optimizer.optimize_all_components(iterations_per_component=5)
        
        successful_optimizations = len([r for r in all_results.values() if r])
        total_components = len(all_results)
        
        print(f"✅ Optimization complete: {successful_optimizations}/{total_components} components optimized")
        
        # Show optimization summary
        print(f"\n📈 OPTIMIZATION SUMMARY:")
        print("-" * 50)
        
        for component_name, result in all_results.items():
            if result:
                param_count = len(result)
                print(f"  {component_name:<20} : {param_count:>2} parameters optimized")
            else:
                print(f"  {component_name:<20} : ❌ optimization failed")
        
        # Generate and save comprehensive report
        print(f"\n💾 SAVING OPTIMIZATION RESULTS:")
        print("-" * 50)
        
        report = optimizer.get_optimization_report()
        optimizer.save_comprehensive_results("demo_optimization_results.json")
        
        print(f"  📊 Total parameters covered: {report['total_parameters']}")
        print(f"  📁 Report saved to: demo_optimization_results.json")
        
        # Show performance improvements
        if 'global_results' in report and 'performance_improvement' in report['global_results']:
            improvements = report['global_results']['performance_improvement']
            print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
            print("-" * 50)
            
            for component, improvement in improvements.items():
                percentage = improvement * 100
                print(f"  {component:<20} : {percentage:>+6.2f}%")
        
        print("\n" + "="*80)
        print("🎉 Comprehensive optimization demo completed successfully!")
        print("🔧 ALL optimizable parameters in MirrorCore-X have been covered")
        print("="*80)
        
        return optimizer, report
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

async def run_targeted_optimization():
    """Run targeted optimization for specific use cases."""
    
    print("\n🎯 TARGETED OPTIMIZATION SCENARIOS")
    print("="*50)
    
    # Create system
    sync_bus, components = await create_mirrorcore_system(dry_run=True)
    optimizer = components['comprehensive_optimizer']
    
    scenarios = {
        "High-Frequency Trading": {
            'categories': ['execution', 'system'],
            'focus': 'Speed and efficiency parameters'
        },
        "Risk Management": {
            'categories': ['risk'],
            'focus': 'Conservative risk parameters'
        },
        "ML/AI Enhancement": {
            'categories': ['ml'],
            'focus': 'Machine learning parameters'
        }
    }
    
    for scenario_name, config in scenarios.items():
        print(f"\n📊 Scenario: {scenario_name}")
        print(f"   Focus: {config['focus']}")
        
        # Optimize categories for this scenario
        total_optimized = 0
        for category in config['categories']:
            results = optimizer.optimize_by_category(category, iterations=3)
            optimized = len([r for r in results.values() if r])
            total_optimized += optimized
            print(f"   {category}: {optimized} components optimized")
        
        print(f"   Total: {total_optimized} components optimized for {scenario_name}")

if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(demo_comprehensive_optimization())
    
    # Run targeted optimization scenarios
    asyncio.run(run_targeted_optimization())
    
    print("\n🚀 All optimization demonstrations completed!")
    print("📖 Check 'demo_optimization_results.json' for detailed results")
