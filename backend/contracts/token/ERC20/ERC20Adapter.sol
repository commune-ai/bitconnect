// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC20/IERC20.sol)

import "interfaces/token/ERC20/IERC20.sol";
import "contracts/token/ERC20/ERC20.sol";

pragma solidity ^0.8.0;

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
 */
contract ERC20Adapter is IERC20 {
    IERC20 public token;

    function connectToken(address _tokenAddress) public {
        token = IERC20(_tokenAddress);

    } 
    /**
     * @dev Returns the amount of tokens in existence.
     */

    function totalSupply() external view override returns (uint256)
    {
        return token.totalSupply();
    }
     
    /**
     * @dev Returns the amount of tokens owned by `account`.
     */
    function balanceOf(address account) external view override returns (uint256)
    {
        return token.balanceOf(account);
    }


    function transfer(address recipient, uint256 amount) external override returns (bool)
    {
        token.transfer(recipient, amount);
    }


    function allowance(address owner, address spender) external view override  returns (uint256)
    {
        token.allowance(owner, spender);
    }


    function approve(address spender, uint256 amount) external override returns (bool)
    {
        token.approve(spender, amount);
    }


    function mint(address account, uint256 amount) public  override  {
        token.mint(account, amount);
    }
 
    function transferFrom(
        address sender,
        address recipient,
        uint256 amount
    ) public override returns (bool) {
        token.transferFrom(sender, recipient, amount);
    }
}