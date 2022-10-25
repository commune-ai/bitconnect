pragma solidity ^0.8.7;
import {TransferHelper} from '@uniswap/v3-periphery/contracts/libraries/TransferHelper.sol';
import {ISushiswapFactory} from 'interfaces/dex/sushiswap/ISushiswapFactory.sol';
import {ISushiswapRouter} from 'interfaces/dex/sushiswap/ISushiswapRouter.sol';

import {DepositableIERC20} from "contracts/token/ERC20/DepositableIERC20.sol";
import {IERC20} from "interfaces/token/ERC20/IERC20.sol";
import "./token/DepositNFTAdapter.sol";
import "contracts/model/base/ModelBase.sol";
import "contracts/utils/Strings.sol";
import { NFTOwnerState, NFTState, NFTContractState} from "./token/Structs.sol";

import {PortfolioState, TokenState} from "./Structs.sol";

// import "contracts/token/ERC20/IERC20.sol";



contract ModelPortfolio is DepositNFTAdapter, ModelBase {
  using Strings for string;
  /* Kovan Addresses */

  event updateStateEvent(PortfolioState state);
  event SwapEvent( string[] tokenSymbolPath, uint256 amountIn);

  event rebalanceEvent(string symbol, int256 valueChange, uint256 initialValue);

  string public hubToken = 'WETH';
  string public baseToken = 'WETH';

  address public wethAddress = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
  DepositableIERC20 wethToken = DepositableIERC20(wethAddress);

  address payable public sushiswapRouterAddress = payable(0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F);
  ISushiswapRouter router = ISushiswapRouter(sushiswapRouterAddress);

  address public sushiswapFactoryAddress  = 0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F;
  ISushiswapFactory factory = ISushiswapFactory(sushiswapFactoryAddress);

  mapping(string=>TokenState) public tokenStates;

  TokenState[] public tokenStatesArray;
  TokenState[][] public tokenStatesHistory;
  string[] public tokenSymbols;
  PortfolioState public state;
  PortfolioState[] public stateHistory;
  // map the block to the state of the portfolio

  uint256 public percentBase = 10000;
  


  constructor(
              string memory _name, 
              string memory _baseTokenSymbol,
              string[] memory _tokenSymbols, address[] memory _tokenAddresses) ModelBase(_name){

    this.setBaseToken(_baseTokenSymbol);
    this.addTokens(_tokenSymbols, _tokenAddresses);
    
  }

  function getTokenStateHistory() view external returns (TokenState[][] memory) {
    TokenState[][] memory tokenStatesHistoryOutput = new TokenState[][](tokenStatesHistory.length);
    for (uint i=0; i<tokenStatesHistory.length; i++ ) {
      tokenStatesHistoryOutput[i]=tokenStatesHistory[i];
    }
    return tokenStatesHistoryOutput;
  }

  function getStateHistory() view external returns (PortfolioState[] memory) {
    PortfolioState[] memory stateHistoryOutput = new PortfolioState[](stateHistory.length);
    for (uint i=0; i<stateHistory.length; i++ ) {
      stateHistoryOutput[i]=stateHistory[i];
    }
    return stateHistoryOutput;
  }


  function setBaseToken(string memory _baseTokenSymbol) public {

    // require(tokenStates[baseTokenSymbol].Address != address(0), 
    //       'Base token isnt tracked fam, use addTokens to add the token');
    state.baseTokenSymbol = _baseTokenSymbol;
  }

  function getTokens() public view returns(string[] memory) {
    return tokenSymbols;
  }



  function addTokens(string[] memory _tokenSymbols, address[] memory _tokenAddresses) public {
    
    for (uint i=0; i<_tokenSymbols.length; i++) {
      
      if (tokenStates[_tokenSymbols[i]].Address == address(0)) {
        tokenStates[_tokenSymbols[i]].Address = _tokenAddresses[i];
        tokenStates[_tokenSymbols[i]].symbol = _tokenSymbols[i];
        tokenSymbols.push(_tokenSymbols[i]);
        tokenStates[_tokenSymbols[i]].index = i;
    }
    }

  }

  function removeTokens(string[] memory _tokenSymbols) public {
      for (uint i=0; i<_tokenSymbols.length; i++) {
        uint tokenSymbolIndex = tokenStates[_tokenSymbols[i]].index;
        delete tokenSymbols[tokenSymbolIndex];
        delete tokenStates[_tokenSymbols[i]];

      }
    }


  function rebalancePortfolio(string[] memory _tokenSymbols, uint256[] memory tokenRatios, bool swapBool) public {
    updateState();

    string[] memory path = new string[](2);
    string memory tokenSymbol;
    int256 tokenRatioChange; 
    int256 relTokenRatioChange;
    for (uint i; i<_tokenSymbols.length; i++) {
        tokenSymbol = _tokenSymbols[i];

        if (tokenSymbol.equals(state.baseTokenSymbol)) {
          continue;
        }

        require(tokenStates[tokenSymbol].Address!=address(0), "The Symbol provided is not in the Portfolio Tokens");

        tokenRatioChange = (int256(tokenRatios[i]) - int256(tokenStates[tokenSymbol].ratio));   

        uint256 tokenBalance = tokenStates[tokenSymbol].balance;





        if (tokenRatioChange<0) { 

          path[0] = tokenSymbol;
          path[1] = hubToken; 

          relTokenRatioChange = (tokenRatioChange*int256(percentBase))/int256(tokenStates[tokenSymbol].ratio);
          tokenStates[tokenSymbol].valueChange = (relTokenRatioChange*int256(tokenBalance))/int(percentBase);


          swap(path, uint256(-tokenStates[tokenSymbol].valueChange));
        } else {

          tokenStates[tokenSymbol].valueChange = (tokenRatioChange*int256(state.marketValue))/int(percentBase);


        }

        emit rebalanceEvent(tokenSymbol, tokenStates[tokenSymbol].valueChange, tokenBalance);

        
    } 

    for (uint i; i<_tokenSymbols.length; i++) {
      tokenSymbol = _tokenSymbols[i];
      if (tokenSymbol.equals(state.baseTokenSymbol)) {
                continue;
              }
      if  (tokenStates[tokenSymbol].valueChange>0) {  

        path[0] = "WETH";
        path[1] =  tokenSymbol; 
        swap(path,  uint(tokenStates[tokenSymbol].valueChange));
      }
    }
    updateState();
  }

  function updateState() public {

    state.marketValue = 0;
    state.blockNumber = block.number;

    for (uint i; i<tokenSymbols.length; i++) {
      string storage tokenSymbol = tokenSymbols[i];
      uint256 tokenBalance = getTokenBalance(tokenSymbol);

      address[] memory path = new address[](2);
      path[0]= tokenStates[tokenSymbol].Address;
      path[1]=  tokenStates[state.baseTokenSymbol].Address;
      tokenStates[tokenSymbol].balance = tokenBalance;
      if (tokenBalance> 0 ) {
        if (path[0] != path[1]) {
          tokenStates[tokenSymbol].value =router.getAmountsOut(tokenBalance, path)[1];
        } else {
          tokenStates[tokenSymbol].value = tokenBalance;
        }

        state.marketValue += tokenStates[tokenSymbol].value;
      }
    }

    for (uint i; i<tokenSymbols.length; i++) {
      tokenStates[tokenSymbols[i]].ratio = (tokenStates[tokenSymbols[i]].value*percentBase)/(state.marketValue+1);
      
      if (i>=tokenStatesArray.length){
        tokenStatesArray.push(tokenStates[tokenSymbols[i]]);
      } else {
        tokenStatesArray[i] = tokenStates[tokenSymbols[i]];
      }
    }

    tokenStatesHistory.push(tokenStatesArray);
    stateHistory.push(state);
    
    emit updateStateEvent(state);

  }

  
  function getTokenBalance(string memory tokenSymbol) public view returns(uint256 balance) {
    IERC20 token = IERC20(tokenStates[tokenSymbol].Address);
    return token.balanceOf(address(this));
  }

  function swapRatio(string[] memory tokenSymbolPath, uint256 amountInRatio, bool swapBool) public {

    // convert ratio (1000 == 100%) to value of token balance
     emit SwapEvent(tokenSymbolPath, amountInRatio);
    
    if (swapBool) {
     uint256 amountIn = (getTokenBalance(tokenSymbolPath[0])*amountInRatio)/percentBase;

    emit SwapEvent(tokenSymbolPath, amountIn);

    swap(tokenSymbolPath, amountIn);  
    }
      }
  function swap(string[] memory tokenSymbolPath, 
                 uint amountIn) public  {

    // address[] memory tokenAddressPath = address()[]

    address[] memory tokenAddressPath = new address[](tokenSymbolPath.length);
    for (uint i; i<tokenAddressPath.length; i++) {
      tokenAddressPath[i]= tokenStates[tokenSymbolPath[i]].Address;
      require(tokenAddressPath[i] != address(0),
         "you passed a null tokenSymbol that doesnt map to the list of addresses");
    }
    
    emit SwapEvent(tokenSymbolPath, amountIn);

    TransferHelper.safeApprove(tokenAddressPath[0], address(router), amountIn);

    uint deadline = block.timestamp + 10000;

    router.swapExactTokensForTokens(amountIn, 0, tokenAddressPath, address(this),deadline);
    }

  function wrapETH() public payable{
    uint ethBalance = address(this).balance;
    require(ethBalance > 0, "No ETH available to wrap");
    DepositableIERC20(tokenStates["WETH"].Address).deposit{ value: ethBalance }();
  }

  function getUserTokenStates(address owner) public view returns (NFTState[] memory){
      
      NFTState[] memory nftStates = depositNFT.getAllOwnerTokenStates(owner);

      for (uint i=0; i<nftStates.length; i++) {
      /* 
      - we want to know how much deposit was done since the token minted
      - this shows how much the market value has changed from when the token was invested
      */ 
      
      uint256 depositValueChange = state.depositValue - nftStates[i].portfolio.depositValue;

      require(depositValueChange>=0, " total deposit value should be greater than token deposit value ");
      // see what the market change was (while removing the future deposits form other users)
      
      uint256 marketValueChangeRatio = ((state.marketValue - depositValueChange)*percentBase)/nftStates[i].portfolio.marketValue;

      nftStates[i].marketValue = (marketValueChangeRatio * nftStates[i].depositValue)/percentBase;

    }

  return nftStates;
  }

  function getUserMarketValue() public view returns ( uint256 ) {
    // get the owner state

    // get the token states that were initially deposited by the owner
    NFTState[] memory nftStates = depositNFT.getAllOwnerTokenStates(msg.sender);

    uint256 userMarketValue = 0;
    for (uint i=0; i<nftStates.length; i++) {
      userMarketValue += nftStates[i].marketValue;
    }
    return userMarketValue;

  }

  function getUserDepositValue() public view returns ( uint256 ) {
    // get the owner state

    // get the token states that were initially deposited by the owner
    NFTState[] memory nftStates = depositNFT.getAllOwnerTokenStates(msg.sender);


    uint256 userDepositValue = 0;
    for (uint i=0; i<nftStates.length; i++) {
      userDepositValue += nftStates[i].depositValue;
    }
    return userDepositValue;

  }

  event WithdrawToken(address to, string tokenSymbol, uint256 balance, uint256 tokenBalance);

  function withdraw(uint256 withdrawRatio) public {

    if (withdrawRatio > percentBase) {
      withdrawRatio=percentBase;
    }

    updateState();

    // get the owner state
    NFTOwnerState memory nftOwnerState = depositNFT.getOwnerState(msg.sender);

    // get the token states that were initially deposited by the owner
    NFTState[] memory nftStates = getUserTokenStates(msg.sender);


    
    for (uint i=0; i<nftStates.length; i++){

      // calculate the entitled ratio of the 
      nftStates[i].marketValue = (nftStates[i].marketValue*withdrawRatio)/percentBase;
      uint256 entitledTreasuryRatio =((nftStates[i].marketValue*percentBase)/state.marketValue);


    for (uint j=0; j<tokenSymbols.length; j++ ) {
      // calculate market value ratio of the depositNFT with portfolio value
      uint256 tokenBalance = tokenStates[tokenSymbols[j]].balance;

      emit WithdrawToken(msg.sender, tokenSymbols[j],entitledTreasuryRatio , tokenBalance);

      if (tokenBalance>0) {

        uint256 tokenValue = tokenStates[tokenSymbols[j]].value;
        // get amount of token balance to withdrawal
        uint256 tokenWithdrawAmount = (tokenBalance*entitledTreasuryRatio)/percentBase;
        uint256 tokenWithdrawValue = (tokenValue*entitledTreasuryRatio)/percentBase;
        
        // transfer amount of tokenStates to msg.sender
        IERC20(tokenStates[tokenSymbols[j]].Address).transfer(msg.sender, tokenWithdrawAmount);
        nftStates[i].depositValue = (((percentBase - withdrawRatio)*nftStates[i].depositValue)/percentBase);
        nftStates[i].marketValue = (((percentBase - withdrawRatio)*nftStates[i].marketValue)/percentBase);


        }
  }
  if (nftStates[i].depositValue==0){
    depositNFT.burn(nftOwnerState.tokenIds[i]);
  }
  else if (nftStates[i].depositValue>0){
        nftStates[i].lastUpdateBlock = block.number;
        depositNFT.updateTokenState(nftOwnerState.tokenIds[i], nftStates[i]);
  }
}

  }

  function deposit() public payable {

        require(msg.value>0, "msg.value is 0 fam");
        wrapETH();
        updateState();


        state.depositValue = state.depositValue + msg.value;
        NFTState memory nftState;

        // when minting, the deposit value and current value are the same
        nftState.depositValue = msg.value;
        nftState.marketValue = nftState.depositValue;

        // use block to track the last timestamp
        nftState.lastUpdateBlock = block.number;
        nftState.portfolio = state;
        depositNFT.mint(msg.sender, nftState);
    }

}