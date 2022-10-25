// // SPDX-License-Identifier: MIT
// // OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

// pragma solidity ^0.8.0;

// import {IERC721Receiver} from "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
// // import {IERC721Metadata} from "@openzeppelin/contracts/token/ERC721/extensions/IERC721Metadata.sol";
// import "@openzeppelin/contracts/utils/Address.sol";
// import "@openzeppelin/contracts/utils/Context.sol";
// import "@openzeppelin/contracts/utils/Strings.sol";
// import "@openzeppelin/contracts/utils/introspection/ERC165.sol";
// import {AccessControlAdapter} from "contracts/utils/access/AccessControlAdapter.sol";
// import {UserState,AssetState} from "contracts/treasury/token/Structs.sol";
// /**
//  * @dev Implementation of https://eips.ethereum.org/EIPS/eip-721[ERC721] Non-Fungible Token Standard, including
//  * the Metadata extension, but not including the Enumerable extension, which is available separately as
//  * {ERC721Enumerable}.
//  */


// contract BaseAssetModule is Context, ERC165, AccessControlAdapter {
//     using Address for address;
//     using Strings for uint256;
//     // Token name
//     NFTContractState public state; 
//     string public name;
//     // Token symbol
//     string public symbol;
//     uint percentBase = 10000;
    

//     // maps user address to assetId to asset ammount
//     mapping(uint256=> mapping(address=>uint256)) public _balances
//     mapping(address=> uint256) public user2index;
//     mapping(string=> uint256) public asset2index;
//     UserState[] public userStates;
//     AssetState[] public assetStates; 

//     mapping(address => mapping(address => uint256)) private _operatorAllowance;
//     mapping(address => mapping(address => bool)) private _operatorApprovals;
//     AssetState public treasuryAssetState;

//     /**
//      * @dev Initializes the contract by setting a `name` and a `symbol` to the token collection.
//      */
//     constructor(string memory _name) {

//         treasuryAssetState.name = _name;
//         trasuryAssetState.asset = address(this);
//     }


//     function deposit(address asset, uint256 balance, address valueOracle,  )


//     function _safeTransferFrom(
//         address from,
//         address to,
//         uint256 id,
//         uint256 amount,
//         bytes memory data
//     ) internal virtual {
//         require(to != address(0), "ERC1155: transfer to the zero address");

//         address operator = _msgSender();

//         _beforeTokenTransfer(operator, from, to, _asSingletonArray(id), _asSingletonArray(amount), data);

//         uint256 fromBalance = _balances[id][from];
//         require(fromBalance >= amount, "ERC1155: insufficient balance for transfer");
//         unchecked {
//             _balances[id][from] = fromBalance - amount;
//         }
//         _balances[id][to] += amount;

//         // emit TransferSingle(operator, from, to, id, amount);

//         _doSafeTransferAcceptanceCheck(operator, from, to, id, amount, data);
//     }


//     /**
//      * @dev See {IERC1155-safeBatchTransferFrom}.
//      */
//     function safeBatchTransferFrom(
//         address from,
//         address to,
//         uint256[] memory ids,
//         uint256[] memory amounts,
//         bytes memory data
//     ) public virtual override {
//         require(
//             from == _msgSender() || isApprovedForAll(from, _msgSender()),
//             "ERC1155: transfer caller is not owner nor approved"
//         );
//         _safeBatchTransferFrom(from, to, ids, amounts, data);
//     }

//     function _safeBatchTransferFrom(
//         address from,
//         address to,
//         uint256[] memory ids,
//         uint256[] memory amounts,
//         bytes memory data
//     ) internal virtual {
//         require(ids.length == amounts.length, "ERC1155: ids and amounts length mismatch");
//         require(to != address(0), "ERC1155: transfer to the zero address");

//         address operator = _msgSender();

//         _beforeTokenTransfer(operator, from, to, ids, amounts, data);

//         for (uint256 i = 0; i < ids.length; ++i) {
//             uint256 id = ids[i];
//             uint256 amount = amounts[i];

//             uint256 fromBalance = _balances[id][from];
//             require(fromBalance >= amount, "ERC1155: insufficient balance for transfer");
//             unchecked {
//                 _balances[id][from] = fromBalance - amount;
//             }
//             _balances[id][to] += amount;
//         }

//         emit TransferBatch(operator, from, to, ids, amounts);

//         _doSafeBatchTransferAcceptanceCheck(operator, from, to, ids, amounts, data);
//     }



//     /**
//      * @dev See {IERC165-supportsInterface}.
//      */


//     function getUserValue(address user) public override view returns (uint256) {
//         // calculate the total user value across assets
//         uint256 user_index = user2index[user];
//         uint256[] memory assetIds = userStates[user_index].assets
//         address[] memory _user_array  = address[](assetIds.length)
//         for (uint i; i<assetIds.length; i++) {
//             _user_array[i] = user
//         }
//         return sumArray(valueOfBatch(_user_array, assetIds))
//     }

//     function getUser(address user) public override view returns(UserState memory){
//         UserState memory _userState=_userStates[user];
//         return _userState;
//     }

//     function getAsset(uint256 id) public view returns(AssetState memory){
//         // get the asset state
//         AssetState memory _asset = assets[id];
//         return _asset;
//     }

//     function filterAssetsByType(uint256[] memory ids,  uint256 type) public override view returns(uint256[] memory){
//         // filters assets by type

//         uint asset_count = 0 ;
//         uint id;
//         for (uint i; i<assetIds.length, i++) {
//             id = ids[i];
//             if (assetStates[ids[i]].type == type))) {
//                 asset_count++;
//             }
//         }
//         uint256[] memory filtered_assets =  uint256[](asset_count)
//         for (i++; i<asset_count, i++) {
//             filtered_assets[i] = assets[i]
//         }

//         return filtered_assets
//     }


    
         

//     function balance2value(uint256 assetId , uint256 balance ) public view returns(uint256) {
//         /**
//         this converts the balance of each asse to the market
//         **/
//         //
//         AssetState storage assetState = assetStates[asseetId];
//         uint256 balance2market_ratio = (assetState.balance*percentBase)/assetState.market_value;
//         uint256 balance_market_value = (balance*balance2market_ratio)/percentBase
//         return balance_market_value;
//     }


//     function balanceOfBatch(address[] memory accounts, uint256[] memory ids)
//         public
//         view
//         virtual
//         override
//         returns (uint256[] memory)
//     {
//         require(accounts.length == ids.length, "ERC1155: accounts and ids length mismatch");

//         uint256[] memory batchBalances = new uint256[](accounts.length);

//         for (uint256 i = 0; i < accounts.length; ++i) {
//             batchBalances[i] = balanceOf(accounts[i], ids[i]);
//         }

//         return batchBalances;
//     }
    
//     function valueOfBatch(address[] memory accounts, uint256[] memory ids)
//         public
//         view
//         virtual
//         override
//         returns (uint256[] memory)
//     {
//         require(accounts.length == ids.length, "ERC1155: accounts and ids length mismatch");

//         uint256[] memory batchValues = new uint256[](accounts.length);

//         for (uint256 i = 0; i < accounts.length; ++i) {
//             batchValues[i] = balance2value(ids[i],balanceOf(accounts[i], ids[i]));
            
//         }

//         return batchValues;
//     }

//     function balanceOf(address account, uint256 id) public view virtual override returns (uint256) {
//         require(account != address(0), "ERC1155: balance query for the zero address");
//         return _balances[id][account];
//     }


//     /**
//      * @dev Returns whether `spender` is allowed to manage `assetId`.
//      *
//      * Requirements:
//      *
//      * - `assetId` must exist.
//      */


//     function addAllowance(address spender, uint256 id, uint256 value) public  returns (bool) {
//         _addAllowance(_msgSender(), spender, id, value)
//     }
//     function _addAllowance(address user, address spender, uint256 id, uint256 value) internal {
//         _operatorAllowance[user][spender][id] = _operatorAllowance[user][spender][id] + value;
//     }


//     function removeAllowance(address spender, uint256 id, uint256 value) public  returns (bool) {
//         _removeAllowance(_msgSender(), spender, id, value);
//     }

//     function _removeAllowance(address user, address spender, uint256 id, uint256 value) internal view virtual returns (bool) {
        
//         uint256 current_allowance = _operatorAllowance[user][spender][id];
//         if (value >= current_allowance) {
//              _operatorAllowance[user][spender][id] = 0;
//         } else {
//             _operatorAllowance[user][spender][id] = _operatorAllowance[user][spender][id] - value;
//         }
//     }


//     function getAllowance(address user, address spender, uint256 id) internal view virtual returns (bool) {
//         return _operatorAllowance[user][spender][id];
//     }


//     function setApprovalForAll(address operator, bool approved) public virtual override {
//         _setApprovalForAll(_msgSender(), operator, approved);
//     }
//     function _setApprovalForAll(
//         address user,
//         address operator,
//         bool approved
//     ) internal virtual {
//         require(user != operator, "ERC721: approve to caller");
//         _operatorApprovals[user][operator] = approved;
//         // emit ApprovalForAll(user, operator, approved);
//     }

//      /**
//      * @dev See {IERC1155-isApprovedForAll}.
//      */
//     function isApprovedForAll(address account, address operator) public view virtual override returns (bool) {
//         return _operatorApprovals[account][operator];
//     }


//     function _beforeTokenTransfer(
//         address operator,
//         address from,
//         address to,
//         uint256[] memory ids,
//         uint256[] memory amounts,
//         bytes memory data
//     ) internal virtual {}

//     function sumArray(uint256[] memory array) public view reteurns(uint256):
//         uint256 total_sum = 0;

//         for (uint i; i< array.length ;  i++) {
//             total_sum = total_sum + array[i];
//         }
//         return total_sum
