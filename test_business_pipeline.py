#!/usr/bin/env python3
"""
Test the new business-first pipeline with the Swedish transcript example
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.multi_agent_summarizer.pipeline import run_pipeline_steps

# Swedish test transcript (abbreviated for testing)
SWEDISH_TEST_TRANSCRIPT = """
[08:07:55.808 - 08:08:10.488 UTC] Det är ju det som behöver preppas och vem som gör vad inför nästa torsdag den 21 och i det. Och jag hängde inte riktigt med i svängen om faciliteringen och tourburenstrål men jag har gjort det att jag vill ha med att lägga in er på det.

[08:08:10.628 - 08:08:28.628 UTC] Det stämmer när man behöver preppa vem som gör vad. Ska vi ta ett varv på var vi står och var vi går? Kopplat till det här avortsamtalet. Nu hade du fått en annan uppgift av Torbjörn också.

[08:08:44.618 - 08:09:02.618 UTC] Vi ska lära oss om preppas och vem som gör vad. Med Jesper igår försökte vi fräscha upp minnet lite. Vad är egentligen syftet och målet med det här? Du och jag har bollat det Mia. Syftet och målet är ju att det är två workshops. Det här är den första.

[08:09:02.438 - 08:09:20.438 UTC] Vi stämmer när det gäller vem som gör vad i workshopen. Vi har någon form av gemensamt föredragande framtidsmag som vi kan leda på. Jag gillar när du säger att vi kan luta oss in i framtiden och omsätta det i praktik i vardagen. Sen har vi pratat lite om artefakter.

[08:12:33.668 - 08:12:51.468 UTC] Vi stämmer över vad som behöver preppas och vem som gör vad. Strategiskt ledning det är Jesper, Ellen, Katja och jag. Sen utenforum då lägger vi till Lisa, Vik och Rebecka Lindblom. Och sen affärsforum då är det Conny och du då. Och sen så är det Jens.

[08:24:32.948 - 08:24:50.948 UTC] Det stämmer när man präpar sig vem som gör vad. Eller så kan det vara så här, nej men vi lutar oss åt en Googleskjut eller en, jag vet inte. Låt oss också ta upp det och konsekvenserna i workshoppandet tänker jag. Men ja, precis. Men som det ser ut nu när vi.

[08:26:36.818 - 08:26:41.818 UTC] De har inte byggt, de har ingen av de metoderna, ramverken, principen.

[08:27:47.198 - 08:28:05.198 UTC] det som behöver kräppas och vem som gör vad. Pilot levererar inte en central sak och det blev så tydligt i Almedalen i den middagen. Så var det hur kollektiv intelligens uppstår.

[09:04:01.928 - 09:04:19.928 UTC] inte det som behöver preppas och vem som gör vad. Ett AI-stöd erbjudande kostar 500 000 kronor att göra. Då måste ni ha ägerskap över det. För ni kan ju inte vara beroende av att jag eller Lotta eller Stefan ska hänga på. Ni måste ju ta ägerskap över att använda det.
"""

def main():
    print("Testing Business-First Pipeline...")
    print("=" * 50)
    
    try:
        # Run the pipeline
        result = run_pipeline_steps(SWEDISH_TEST_TRANSCRIPT)
        
        print("Pipeline completed successfully!")
        print("\n=== PIPELINE OUTPUTS ===\n")
        
        # Print each layer
        for key, content in result.items():
            if key.endswith('_md') and content:
                print(f"## {key.upper()}")
                print("-" * 40)
                print(content[:500] + "..." if len(content) > 500 else content)
                print("\n")
        
        # Save full output to file
        with open('/tmp/pipeline_test_output.md', 'w', encoding='utf-8') as f:
            f.write(result.get('full_md', 'No full output'))
        
        print(f"Full output saved to: /tmp/pipeline_test_output.md")
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()