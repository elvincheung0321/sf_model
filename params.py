import stable_retro

game_name = "StreetFighterIISpecialChampionEdition-Genesis-v0"

# Retrieve and print all available starting states for the game
try:
    available_states = stable_retro.data.list_states(game_name)
    print(f"Available states for {game_name}:")
    for state in available_states:
        print(f" - {state}")
except FileNotFoundError:
    print(f"Game '{game_name}' not found. Check stable_retro.data.list_games().")
