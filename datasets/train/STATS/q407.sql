select  count(*) from comments as c,  		postHistory as ph,          votes as v,  		users as u where u.Id  = v.UserId  	and v.UserId = ph.UserId  	and ph.UserId =c.UserId  AND c.Score=0  AND ph.CreationDate>='2010-10-18 21:55:09'::timestamp  AND u.Reputation>=1  AND u.Views>=0  AND u.Views<=126  AND u.DownVotes<=16  AND u.CreationDate<='2014-07-27 08:51:25'::timestamp;