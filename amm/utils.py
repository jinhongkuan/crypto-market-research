from loguru import logger

def parse_filename_fee_switch(file: str) -> tuple[str, str, str, str]:
    filename_parts = file.split('_')
    try:
        chain = filename_parts[1]
        token_pair = f"{filename_parts[3]}_{filename_parts[4]}"
        fee_tier = filename_parts[5]
        fee_switch = filename_parts[6]
    except (IndexError, ValueError) as e:
        logger.warning(f"Could not parse metadata from filename {file}, skipping: {str(e)}")
        return None
            
    return chain, token_pair, fee_tier, fee_switch