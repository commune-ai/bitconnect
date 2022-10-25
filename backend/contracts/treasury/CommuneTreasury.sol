// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;

import {IERC721Receiver} from "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
// import {IERC721Metadata} from "@openzeppelin/contracts/token/ERC721/extensions/IERC721Metadata.sol";
import "@openzeppelin/contracts/utils/Address.sol";
import "@openzeppelin/contracts/utils/Context.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/utils/introspection/ERC165.sol";
import {AccessControlAdapter} from "contracts/utils/access/AccessControlAdapter.sol";
import {UserState,AssetState} from "contracts/treasury/Structs.sol";
import {AssetToken} from "contracts/treasury/token/AssetToken.sol";
/**
 * @dev Implementation of https://eips.ethereum.org/EIPS/eip-721[ERC721] Non-Fungible Token Standard, including
 * the Metadata extension, but not including the Enumerable extension, which is available separately as
 * {ERC721Enumerable}.
 */

contract CommuneTreasury is Context, ERC165, AccessControlAdapter {
    using Address for address;
    using Strings for uint256;
    // Token name
    // Token symbol
    uint percentBase = 10000;
    
    // maps user address to assetId to asset ammount
    
    // asset to user to balance
    mapping(address=> mapping(address=>uint256)) public _balances;
    mapping(address=> uint256) public asset2index;
    
    mapping(address=> address[]) public user2assets;
    address[] public users;


    AssetState[] public assetStates; 

    mapping(address => mapping(address => uint256)) private _operatorAllowance;
    mapping(address => mapping(address => bool)) private _operatorApprovals;
    AssetState public treasuryAssetState;

    /**
     * @dev Initializes the contract by setting a `name` and a `symbol` to the token collection.
     */
     AssetToken assetToken;




    constructor(string memory _name, string memory _symbol) {

        // treasuryAssetState.name = _name;
        treasuryAssetState.asset = address(this);
        assetToken = new AssetToken(address(this), 'treasury');
    }

    function assetExists(address asset) public view returns(bool) {
        if (assetStates.length == 0) {
            return false;
        }
        return (assetStates[asset2index[asset]].asset == asset);
    }


    function removeAsset(address asset) internal {

        // dummy value for oracle
        uint256 assetIndex = asset2index[asset];
        assetStates[assetIndex].valueOracle = address(0);

        delete assetStates[assetIndex];
        delete asset2index[asset];

    }

    function removeUserAsset(address user, address asset) internal {

        // remove asset from user
        for (uint i; i < user2assets[user].length; i++) {
            if (user2assets[user][i] == asset) {
                delete user2assets[user][i];
            }
            
        }
        }


    function userHasAsset(address user, address asset) internal returns(bool){
        bool hasAsset = false;
        for (uint i; i < user2assets[user].length; i++) {
            if (user2assets[user][i] == asset) {
                hasAsset=true;
                break;
            }
            
        }
        return hasAsset;
    }

    function addUserAsset(address user, address asset) internal {

        // remove asset from user

        bool hasAsset = false;
        for (uint i; i < user2assets[user].length; i++) {
            if (user2assets[user][i] == asset) {
                hasAsset=true;
                break;
            }
            
        }
        // if the user does not have the asset
        if (!hasAsset) {
            user2assets[user].push(asset);
        }
        }


    function getAssetOracleValue(address asset) internal returns (uint256) {
        address oracle = assetStates[asset2index[asset]].valueOracle;

        // get the oracle value per unit
        return 2;
    }

    function updateAssets() internal {
        for (uint i; i<assetStates.length; i++) {
            uint256 value_per_unit = getAssetOracleValue(assetStates[i].asset);
            assetStates[i].value = value_per_unit* assetStates[i].balance;
            assetStates[i].lastUpdateBlock = block.number;
            treasuryAssetState.value = treasuryAssetState.value + assetStates[i].value; 
        }
        
    }

    function addAsset(address asset, uint256 balance, string memory name, string memory mode, bool liquid ) public {
        AssetState memory _assetState;
        _assetState.asset = asset;
        _assetState.name = name;
        _assetState.balance = balance; 
        _assetState.mode = mode; 
        _assetState.liquid = liquid; 

        assetStates.push(_assetState);
    }



    function deposit(address asset, uint256 balance, string memory name, string memory mode, bool liquid  ) public {
        if (!assetExists(asset)) {
            addAsset(asset, balance, name, mode, liquid);
        }
        address account = _msgSender();
        uint256 assetIndex = asset2index[asset];
        _balances[asset][account] += balance;
        addUserAsset(account, asset);
        updateAssets();
    }



    function withdraw(address asset, uint256 balance) public {
        uint256 assetIndex = asset2index[asset];
        AssetState storage assetState = assetStates[assetIndex];
        balance = max(assetState.balance, balance);
        assetState.balance = assetState.balance - balance;

        // if the state has 0 balance, delete the asset
        if ( assetState.balance == 0) {
            removeUserAsset(_msgSender(), asset);
        }
    }


    function _safeTransferFrom(
        address from,
        address to,
        address asset,
        uint256 amount,
        bytes memory data
    ) internal virtual {
        require(to != address(0), "ERC1155: transfer to the zero address");

        address operator = _msgSender();

        address[] memory asset_array = new address[](1);
        asset_array[0] = asset;

        uint256[] memory amount_array = new uint256[](1);
        amount_array[0] = amount;

        _beforeTokenTransfer(operator, from, to, asset_array, amount_array, data);

        uint256 fromBalance = _balances[asset][from];
        require(fromBalance >= amount, "ERC1155: insufficient balance for transfer");
        unchecked {
            _balances[asset][from] = fromBalance - amount;
        }
        _balances[asset][to] += amount;

        // emit TransferSingle(operator, from, to, asset, amount);

        // _doSafeTransferAcceptanceCheck(operator, from, to, asset, amount, data);
    }


    /**
     * @dev See {IERC1155-safeBatchTransferFrom}.
     */
    function safeBatchTransferFrom(
        address from,
        address to,
        address[] memory assets,
        uint256[] memory amounts,
        bytes memory data
    ) public virtual {
        require(
            from == _msgSender() || isApprovedForAll(from, _msgSender()),
            "ERC1155: transfer caller is not owner nor approved"
        );
        _safeBatchTransferFrom(from, to, assets, amounts, data);
    }

    function _safeBatchTransferFrom(
        address from,
        address to,
        address[] memory assets,
        uint256[] memory amounts,
        bytes memory data
    ) internal virtual {
        require(assets.length == amounts.length, "ERC1155: assets and amounts length mismatch");
        require(to != address(0), "ERC1155: transfer to the zero address");

        address operator = _msgSender();

        _beforeTokenTransfer(operator, from, to, assets, amounts, data);

        for (uint256 i = 0; i < assets.length; ++i) {
            address asset = assets[i];
            uint256 amount = amounts[i];

            uint256 fromBalance = _balances[asset][from];
            require(fromBalance >= amount, "ERC1155: insufficient balance for transfer");
            unchecked {
                _balances[asset][from] = fromBalance - amount;
            }
            _balances[asset][to] += amount;
        }

        // emit TransferBatch(operator, from, to, assets, amounts);

    }

    /**
     * @dev See {IERC165-supportsInterface}.
     */

    function balanceOfBatch(address user) public returns(address[] memory, uint256[] memory) 
    {
        address[] memory assets = user2assets[user];
        uint256[] memory assetBalances = new uint256[](assets.length);
        for (uint i;i<assets.length; i++) {
            assetBalances[i] = assetStates[asset2index[assets[i]]].balance;
        }
        return (assets, assetBalances);
    }

    function getUserValue(address user) public view returns (uint256) {
        // calculate the total user value across assets
        address[] memory assets = user2assets[user];
        address[] memory _user_array  = new address[](assets.length);
        for (uint i; i<assets.length; i++) {
            _user_array[i] = user;
        }
        return sumArray(valueOfBatch(_user_array, assets));
    }

    // function getUser(address user) public override view returns(UserState memory){
    //     UserState memory _userState=_userStates[user];
    //     return _userState;
    // }


    function getAssetStates() public view returns(AssetState[] memory){
        // get the asset state
        AssetState[] memory _assets = assetStates;
        return _assets;
    }

    function filterAssetsByMode(address[] memory assets,  string memory mode) public view returns(address[] memory){
        // filters assets by mode

        uint asset_count = 0 ;
        for (uint i; i<assets.length; i++) {
            uint assetIndex = asset2index[assets[i]];
            if (keccak256(abi.encodePacked((assetStates[assetIndex].mode))) == keccak256(abi.encodePacked((mode)))){
                asset_count++;
            }
        }
        address[] memory filtered_assets =  new address[](asset_count);
        
        for (uint i; i<asset_count; i++) {
            filtered_assets[i] = assets[i];
        }

        return filtered_assets;
    }

    function balance2value(address asset , uint256 balance ) public view returns(uint256) {
        /**
        this converts the balance of each asse to the market
        **/
        //

        uint256 assetId = asset2index[asset];
        AssetState storage assetState = assetStates[assetId];
        uint256 balance2market_ratio = (assetState.balance*percentBase)/assetState.value;
        uint256 balance_market_value = (balance*balance2market_ratio)/percentBase;
        return balance_market_value;
    }


    function balanceOfBatch(address[] memory accounts, address[] memory assets)
        public
        view
        virtual
        returns (uint256[] memory)
    {
        require(accounts.length == assets.length, "ERC1155: accounts and assets length mismatch");

        uint256[] memory batchBalances = new uint256[](accounts.length);

        for (uint256 i = 0; i < accounts.length; ++i) {
            batchBalances[i] = balanceOf(accounts[i], assets[i]);
        }

        return batchBalances;
    }
    
    function valueOfBatch(address[] memory accounts, address[] memory assets)
        public
        view
        virtual
        
        returns (uint256[] memory)
    {
        require(accounts.length == assets.length, "ERC1155: accounts and assets length mismatch");

        uint256[] memory batchValues = new uint256[](accounts.length);

        for (uint256 i = 0; i < accounts.length; ++i) {
            batchValues[i] = balance2value(assets[i],balanceOf(accounts[i], assets[i]));
            
        }

        return batchValues;
    }

    function balanceOf(address account, address asset) public view virtual returns (uint256) {
        require(account != address(0), "ERC1155: balance query for the zero address");
        return _balances[asset][account];
    }

    function valueOf(address account, address asset) public view virtual returns (uint256) {
        require(account != address(0), "ERC1155: balance query for the zero address");
        return balance2value(asset, _balances[asset][account] );
    }


    function _setApprovalForAll(
        address user,
        address operator,
        bool approved
    ) internal virtual {
        require(user != operator, "ERC721: approve to caller");
        _operatorApprovals[user][operator] = approved;
        // emit ApprovalForAll(user, operator, approved);
    }

     /**
     * @dev See {IERC1155-isApprovedForAll}.
     */
    function isApprovedForAll(address account, address operator) public view virtual returns (bool) {
        return _operatorApprovals[account][operator];
    }


    function setApprovalForAll(address operator, bool approved) public virtual {
        _setApprovalForAll(_msgSender(), operator, approved);
    }

    function _beforeTokenTransfer(
        address operator,
        address from,
        address to,
        address[] memory assets,
        uint256[] memory amounts,
        bytes memory data
    ) internal virtual {}

    function sumArray(uint256[] memory array) public view returns(uint256){
        uint256 total_sum = 0;

        for (uint i; i< array.length ; i++) {
            total_sum = total_sum + array[i];
        }
        return total_sum;
    }

    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a >= b ? a : b;
    }

}




    /**
     * @dev Returns whether `spender` is allowed to manage `assetId`.
     *
     * Requirements:
     *
     * - `assetId` must exist.
     */


    // function addAllowance(address spender, uint256 asset, uint256 value) public  returns (bool) {
    //     _addAllowance(_msgSender(), spender, asset, value)
    // }
    // function _addAllowance(address user, address spender, uint256 asset, uint256 value) internal {
    //     _operatorAllowance[user][spender][asset] = _operatorAllowance[user][spender][asset] + value;
    // }


    // function removeAllowance(address spender, uint256 asset, uint256 value) public  returns (bool) {
    //     _removeAllowance(_msgSender(), spender, asset, value);
    // }

    // function _removeAllowance(address user, address spender, uint256 asset, uint256 value) internal view virtual returns (bool) {
        
    //     uint256 current_allowance = _operatorAllowance[user][spender][asset];
    //     if (value >= current_allowance) {
    //          _operatorAllowance[user][spender][asset] = 0;
    //     } else {
    //         _operatorAllowance[user][spender][asset] = _operatorAllowance[user][spender][asset] - value;
    //     }
    // }


    // function getAllowance(address user, address spender, uint256 asset) internal view virtual returns (bool) {
    //     return _operatorAllowance[user][spender][asset];
    // }
