#%%
from agent_model import agent_journal
import matplotlib.pyplot as plt
import seaborn as sns

model = agent_journal.UnivModel(100,5,100,class_periods=3,class_size=10,majors=True)
for _ in range(model.class_periods):
    model.step()

plt.imshow(model.contactjournal)
#sns.set_context("talk")
plt.xlabel("Agent ID")
plt.ylabel("Agent ID")
plt.title("Agent Contact Counts")
plt.colorbar()
plt.savefig("figures/figure2")

# %%
