select  count(*) from comments as c,          postHistory as ph,          users as u where u.Id = c.UserId 	and c.UserId = ph.UserId  AND c.Score=0  AND ph.CreationDate>='2010-12-06 14:53:21'::timestamp  AND u.Reputation>=1  AND u.Reputation<=213  AND u.Views>=0  AND u.Views<=79;