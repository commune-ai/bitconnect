from commune import Module
import streamlit as st


class ModelContract(Module):
    def __init__(self):
        Module.__init__(self)
        self.contract_manager = self.launch('web3.contract')
        self.contract_manager.set_account('alice')
        self.contract = self.contract_manager.deploy_contract('token.ERC20.ModelToken', new=False)
        self.account = self.contract.account
        self.web3 = self.contract.web3
        self.dataset = self.launch('huggingface.dataset')


    def mint(self, amount=1000):
        return self.contract.mint(self.account.address, amount)


    def add_stake(self, amount):
        self.contract.approve(self.contract.address,amount)
        self.contract.add_stake(amount)

    def remove_stake(self, amount):
        self.contract.remove_stake(amount)

    def get_stake(self):
        self.get_stake(self.account.address)

    def set_votes(self, accounts, votes):
        self.contract.set_votes(votes, accounts)

    @classmethod
    def streamlit_demo(self):
        self = ModelContract()
        amount = 10000
        st.write(self.mint(10000))
        st.write(self.add_stake(1000))
        st.write(self.set_votes([self.account.address], [100]))

    @classmethod
    def streamlit(cls):
        cls.streamlit_demo()
        





if __name__ == '__main__':
    ModelContract.run()
