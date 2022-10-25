// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.7;

import "contracts/utils/access/AccessControlAdapter.sol";
import "contracts/explain/ExplainManagerAdapter.sol";
import "contracts/token/ERC20/ERC20Adapter.sol";
import "contracts/token/ERC20/ERC20.sol";
import "contracts/utils/context/Context.sol";
import "contracts/utils/Strings.sol";
import {IPNSObject} from "contracts/ipfs/Structs.sol";
import {ModelURI, ModelState} from "contracts/model/base/Structs.sol";
// pointers ot off chain resources that associate with the model


contract ModelBase {

    string public name;
    ModelState public modelState;

    constructor(string memory _name) {
        name = _name;
    }
    // function processPayment(address userAddress) public virtual override payable {
    // }

    function setExplain(string memory uri) public  {
        modelState.uri.explain = uri;
    }
    function setModel(string memory uri) public  {
        modelState.uri.model = uri;
    }


    function setState(ModelState memory _modelState) public {
        modelState= _modelState; 
    }


    function getModel() public view returns (string memory){
        return modelState.uri.model;
    }
    function getExplain() public view returns (string memory){
        return modelState.uri.explain;
    }
    function getState() public view returns(ModelState memory){
        return modelState;
    }
}
