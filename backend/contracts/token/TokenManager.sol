// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.7;

/*
@title myVault
@license GNU GPLv3
@author James Bachini
@notice A vault to automate and decentralize a long term donation strategy
*/
// import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import "contracts/token/TokenManagerBase.sol";


// interface IERC20Extented is IERC20 {
//     function decimals() view returns (uint8);
// }


contract TokenManager is TokenManagerBase{

    constructor(bytes32[] memory _token_symbol_list, address[] memory _token_address_list) {
        addTokens(_token_symbol_list, _token_address_list);
    }

    function addToken(bytes32 _tokenSymbol, address _tokenAddress) public  {

            // map the symbol to the address 
            symbol_address_map[_tokenSymbol] = _tokenAddress;

            // map the symbol to the token
            symbol_token_map[_tokenSymbol] = ERC20(_tokenAddress);

    }

    function addTokens(bytes32[] memory _token_symbol_list, address[] memory _token_address_list ) public {

        require(_token_symbol_list.length==_token_address_list.length, "token_address_list and token_symbol_list must equal");
        for (uint i=0;i<_token_symbol_list.length; i++)
        {
            addToken( _token_symbol_list[i], _token_address_list[i]);
            token_count ++;
        }

    }


    
}