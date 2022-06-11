### Glossary:
 
- **Preflop (PF)**: The first betting round(round 0), after dealing hole cards to players
- **Flop (F)**: Second Betting round(round 1), when first three board card(also called flop!) are shown. 
- **Turn (T)**: Third Betting Round(round 2), when 4'th board card(also called turn!) is shown
- **River (R)**: Fourth and last betting round( round 3), when 5'th (last) board card(aslo card river) is shown. 
- **Big Blind(BB)**: The blind forced bet that player at Big Blind Position must put in the pot before seeing his hand.
It is actually unit of poker table currency
- **Showdown(SW)**: After last betting round, if there are more than one player in the pot

### Filters:
- player name
- Position, relative position, Number of Players acted Before and to act after
- BB size
- action amount of bet or raise or call
- Preflop(PF), Flop(F), Turn(T), River(R)
- SPR: Stack To Pot Ratio
- Action History
- HoleCards
- Flop, Turn and River Cards
- Player Hands
## Population Stat


## Player Stats
### Number  Of Played Hands
- Number of dealt hands to player
- Number of hands that player put any money in the pot(no matter lost or won)
- Sum of 1/n for Preflops seen by player, where n is number of players in the pot. Also same for all Flop, Turn and 
   River seen  
- Number of hands with a blind post
- Number of hands that player voluntarily put money in  pot
- VPIP = (Number of hands that voluntarily put money in  pot )/Number of dealt hands
- Number of PF, F, T, R seen
- Number of hands folded on PF, F, T, R, Total
- Number of non showdown wins on PF, F, T, R, Total
- Number of showdown wins on river
- Total number of hands won
- Total number of hands that won money

### Winning Amounts and Investment amounts
- Data of Amount won on each hand (Toman and BB)
- Total Amount Won (Toman and BB )
- Total Amount won  at Showdown(Toman and BB)
- Total Amount won without showdown(Toman and BB)
- Data of Amount Invested at each hand(Toman and BB)
- Total Investments in all hands(Toman and BB)
- Data of rate of return on each hand: (Amount won)/(Amount Invested in Hand)
- Total Rate Of Return: (Total Amount Won)/(Total Investment in all hands)
- Data of total site rake at each hand(Toman and BB)
- Data of total rake paid by player
- Winrate per hand: (Total Amount Won) / (Number of dealt hands)
- Winrate (per 100 hands): 
   100*(Total Amount Won) / (Number of dealt hands)
- Total amount won after a given action(hand history)  
  possible actions: bluffing, river bet, flop cbet, steal
- Data of EV adjusted amount won



### Statistical Action Frequencies
#### Preflop action when folded to:
  - **RFI_N**: Number of raises as the first player who put money in the pot PF
  - **LFI_N**: Number of calls as the first player who put money in the pot PF
  - **FI_N**: _Number_ of opportunity for Raise First In or Limp First In  PF
  - **RFI_P** = NRFI / NFI
  - **LFI_P** = NLFI / NFI
  - **Steal_N** = number of preflop raise first in BTN position
  - **Steal_P** = NSteal / (NFI in BTN)
 
#### Preflop action when facing LFI or RFI:

  - **ISO_N**: Number of raises when facing an Limp First In preflop
  - **ISO_P**
  - **Sqz_N**: Number of raises when facing an Raise First In + some calls preflop
  - **Sqz_P** = (Sqz_N) / (Number of Sqz Opportunities)
  - **ColdCall_N**:
  - **ColdCall_P**:
  - 
#### Preflop, Flop, Turn, River action frequencies:
  
 - **Bet_N**: Total Number of Bets
 - **Bet_P**: (Bet_N)/(Number of Bet Opportunities)
 - **Raise_N**: Number of raises when facing a Bet
 - **Raise_P**: (Raise_N) / (Number of Raise opportunities)
 - **3Bet_N**: Number of re-raises when facing a Raise(2Bet)
 - **3Bet_P**: (3Bet_N) / (Number of 3 Bet Opportunities)
 - **4Bet_N**: Number of re-re-raises when facing a 3Bet(re-raise)
 - **4Bet_P**: (4Bet) / (Number of 4Bet Opportunities)
 - **CallVsBet_N**: 
 - **CallVsBet_P**:
 - **FoldVsBet_N**:
 - **FoldVsBet_P**:
 - **CallVsRaise_N**:
 - **CallVsRaise_P**:
 - **FoldVsRaise_N**:
 - **FoldVsRaise_P**:
 - **CallVs3Bet_N**:
 - **CallVs3Bet_P**:
 - **FoldVs3Bet_N**:
 - **FoldVs3Bet_P**:
 - **CallVs4Bet_N**:
 - **CallVs4Bet_P**:
 - **FoldVs4Bet_N**:
 - **FoldVs4Bet_P**:
 - **AllRaises_N**:
 - **AllRaises_P**:
 - **AllCalls_N**:
 - **AllCalls_P**:
 
##### Special Cation Frequencies:
 - **CBet_N**: Number of hands that player is the aggressor of all previous street(betting round) and bet in current street
 - **FlopCbet_N**: number of hands that Cbet on flop as a Preflop aggressor
 - **TurnCbet_N**:
 - **TurnCbet_P**
 - **RiverCbet_N**:
 - **RiverCbet_P**:
 - **FoldVsFlopCBet**:
 - **FoldVsTurnCbet**
 - **FoldVsRiverCbet**
 - **AttemptToSteal**: Betting when checked to as last player to act in that street(BTN for PF)
 - **StealAndFold**: Folding at any point in the hand after stealing
 - **StealWinSD**: 
































































