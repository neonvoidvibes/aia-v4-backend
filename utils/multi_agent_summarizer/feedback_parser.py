"""
Reality Check Feedback Parser
Extracts agent-specific feedback from reality check output to enable targeted refinements.
"""

import re
from typing import Dict, Optional


def parse_reality_check_feedback(reality_check_md: str) -> Dict[str, str]:
    """
    Parse reality check markdown and extract feedback relevant to each agent layer.
    
    Returns dict with keys: context, business_reality, organizational_dynamics, 
    strategic_implications, next_actions
    """
    
    feedback = {
        "context": "",
        "business_reality": "", 
        "organizational_dynamics": "",
        "strategic_implications": "",
        "next_actions": ""
    }
    
    # Extract sections using markdown headers and content
    sections = {}
    current_section = None
    current_content = []
    
    lines = reality_check_md.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Detect section headers
        if line.startswith('### '):
            # Save previous section
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            
            # Start new section
            current_section = line[4:].strip().lower()
            current_content = []
        
        elif line.startswith('# ') or line.startswith('## '):
            # Save previous section if any
            if current_section:
                sections[current_section] = '\n'.join(current_content)
                current_section = None
                current_content = []
        
        else:
            # Add content to current section
            if current_section and line:
                current_content.append(line)
    
    # Save final section
    if current_section:
        sections[current_section] = '\n'.join(current_content)
    
    # Map sections to agent feedback
    feedback["context"] = _extract_context_feedback(sections)
    feedback["business_reality"] = _extract_business_reality_feedback(sections)
    feedback["organizational_dynamics"] = _extract_org_dynamics_feedback(sections)
    feedback["strategic_implications"] = _extract_strategic_feedback(sections)
    feedback["next_actions"] = _extract_actions_feedback(sections)
    
    return feedback


def _extract_context_feedback(sections: Dict[str, str]) -> str:
    """Extract feedback relevant to business context."""
    relevant_content = []
    
    # Look for context/business context related feedback
    for section_name, content in sections.items():
        if any(keyword in section_name for keyword in ['business context', 'context', 'meeting purpose']):
            relevant_content.append(f"**{section_name.title()}**: {content}")
    
    # Also extract general accuracy feedback that might apply to context
    accuracy_section = sections.get('accuracy check', '')
    if accuracy_section:
        # Extract context-related parts
        context_feedback = _extract_layer_specific_feedback(accuracy_section, ['context', 'business context', 'meeting'])
        if context_feedback:
            relevant_content.append(f"**Accuracy Feedback**: {context_feedback}")
    
    return '\n\n'.join(relevant_content) if relevant_content else "No specific context feedback provided."


def _extract_business_reality_feedback(sections: Dict[str, str]) -> str:
    """Extract feedback relevant to business reality layer."""
    relevant_content = []
    
    # Look for business reality related feedback
    for section_name, content in sections.items():
        if any(keyword in section_name for keyword in ['business reality', 'layer 1', 'decisions', 'tasks', 'meeting facts']):
            relevant_content.append(f"**{section_name.title()}**: {content}")
    
    # Extract from accuracy check
    accuracy_section = sections.get('accuracy check', '')
    if accuracy_section:
        reality_feedback = _extract_layer_specific_feedback(accuracy_section, ['business reality', 'layer 1', 'decisions', 'tasks'])
        if reality_feedback:
            relevant_content.append(f"**Accuracy Feedback**: {reality_feedback}")
    
    # Extract from missing content
    missing_section = sections.get('missing critical content', '')
    if missing_section:
        relevant_content.append(f"**Missing Content**: {missing_section}")
    
    return '\n\n'.join(relevant_content) if relevant_content else "No specific business reality feedback provided."


def _extract_org_dynamics_feedback(sections: Dict[str, str]) -> str:
    """Extract feedback relevant to organizational dynamics."""
    relevant_content = []
    
    # Look for org dynamics related feedback
    for section_name, content in sections.items():
        if any(keyword in section_name for keyword in ['organizational', 'dynamics', 'layer 2', 'patterns', 'communication']):
            relevant_content.append(f"**{section_name.title()}**: {content}")
    
    # Extract from accuracy check
    accuracy_section = sections.get('accuracy check', '')
    if accuracy_section:
        dynamics_feedback = _extract_layer_specific_feedback(accuracy_section, ['organizational', 'dynamics', 'layer 2', 'patterns'])
        if dynamics_feedback:
            relevant_content.append(f"**Pattern Validity Feedback**: {dynamics_feedback}")
    
    return '\n\n'.join(relevant_content) if relevant_content else "No specific organizational dynamics feedback provided."


def _extract_strategic_feedback(sections: Dict[str, str]) -> str:
    """Extract feedback relevant to strategic implications."""
    relevant_content = []
    
    # Look for strategic related feedback
    for section_name, content in sections.items():
        if any(keyword in section_name for keyword in ['strategic', 'implications', 'layer 3', 'business impact']):
            relevant_content.append(f"**{section_name.title()}**: {content}")
    
    # Extract from accuracy check
    accuracy_section = sections.get('accuracy check', '')
    if accuracy_section:
        strategic_feedback = _extract_layer_specific_feedback(accuracy_section, ['strategic', 'implications', 'layer 3'])
        if strategic_feedback:
            relevant_content.append(f"**Strategic Relevance Feedback**: {strategic_feedback}")
    
    return '\n\n'.join(relevant_content) if relevant_content else "No specific strategic implications feedback provided."


def _extract_actions_feedback(sections: Dict[str, str]) -> str:
    """Extract feedback relevant to next actions."""
    relevant_content = []
    
    # Look for actions related feedback
    for section_name, content in sections.items():
        if any(keyword in section_name for keyword in ['actions', 'layer 4', 'actionable', 'next steps']):
            relevant_content.append(f"**{section_name.title()}**: {content}")
    
    # Extract from accuracy check
    accuracy_section = sections.get('accuracy check', '')
    if accuracy_section:
        actions_feedback = _extract_layer_specific_feedback(accuracy_section, ['actions', 'layer 4', 'feasibility'])
        if actions_feedback:
            relevant_content.append(f"**Action Feasibility Feedback**: {actions_feedback}")
    
    return '\n\n'.join(relevant_content) if relevant_content else "No specific next actions feedback provided."


def _extract_layer_specific_feedback(content: str, keywords: list) -> Optional[str]:
    """Extract sentences from content that mention any of the keywords."""
    sentences = content.split('.')
    relevant_sentences = []
    
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            relevant_sentences.append(sentence.strip())
    
    return '. '.join(relevant_sentences) + '.' if relevant_sentences else None